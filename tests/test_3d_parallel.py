"""Hunyuan3D 2.0, Hunyuan3D 2.1 and TRELLIS.2 / Pixal3D through Raylight's FSDP, USP and CFG paths, checked on every
rank against ComfyUI's own unsharded model.

    torchrun --nproc_per_node=2 tests/test_3d_parallel.py
    torchrun --nproc_per_node=3 tests/test_3d_parallel.py   # odd world: every sequence needs padding
    torchrun --nproc_per_node=4 tests/test_3d_parallel.py   # adds USP x CFG on top of FSDP

Runs on CUDA with NCCL when there is a card, otherwise on CPU with gloo. Under plain pytest there is no process
group and the tests skip.
"""
import contextlib
import os
import shutil
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

# torchrun runs this file directly, without conftest: find Raylight's src and the ComfyUI checkout it sits in
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CUDA = torch.cuda.is_available()
if CUDA and "LOCAL_RANK" in os.environ:
    # before comfy picks its device at import, so every rank opens its context on its own card
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
if not CUDA:
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import safetensors.torch
from torch.distributed.device_mesh import init_device_mesh

import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.sample
import comfy.sd
import comfy.supported_models
from comfy.ldm.hunyuan3d.model import Hunyuan3Dv2
from comfy.ldm.hunyuan3dv2_1.hunyuandit import HunYuanDiTPlain
from comfy.ldm.trellis2.model import Trellis2

import raylight.diffusion_models.trellis2.xdit_context_parallel as trellis2_usp
import raylight.diffusion_models.utils as diffusion_utils
import raylight.distributed_modules.cfg_utils as cfg_utils
from raylight.comfy_dist.model_patcher import FSDPModelPatcher
from raylight.comfy_dist.sd import fsdp_load_diffusion_model
from raylight.comfy_extra_dist.nodes_guidance import RayCFGOverride
from raylight.diffusion_models.hunyuan3d.xdit_cfg_parallel import cfg_parallel_forward_wrapper
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry
from raylight.distributed_modules.usp import USPInjectRegistry

MODEL_OPTIONS = {"dtype": torch.float32}
# The Trellis2 checkpoint layout always detects the full 4B configuration; these shrink it, the way a smaller model
# config would.
TRELLIS2_SIZE = {"model_channels": 64, "cond_channels": 32, "num_blocks": 2, "num_heads": 4, "mlp_ratio": 2.0}
TRELLIS2_PROJ = {"image_attn_mode_structure": "proj", "proj_in_channels_structure": 1024,
                 "image_attn_mode_shape": "proj", "proj_in_channels_shape": 2048}


def _random_state_dict(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randn(param.shape, generator=g) * 0.05 for name, param in model.state_dict().items()}


def _hunyuan3d_checkpoint():
    model = Hunyuan3Dv2(in_channels=8, context_in_dim=24, hidden_size=64, num_heads=16, depth=2, depth_single_blocks=2,
                        guidance_embed=True, operations=comfy.ops.disable_weight_init, dtype=torch.float32)
    return _random_state_dict(model, seed=1)


def _hunyuan3d21_checkpoint():
    # ComfyUI's detection fixes heads, context width and the MoE layout, so this is the 2.1 architecture at width 64
    model = HunYuanDiTPlain(in_channels=8, hidden_size=64, context_dim=1024, depth=4, num_heads=16, qkv_bias=False,
                            guidance_cond_proj_dim=None, operations=comfy.ops.disable_weight_init, dtype=torch.float32)
    return _random_state_dict(model, seed=2)


def _trellis2_checkpoint(proj=False):
    model = Trellis2(resolution=64, init_txt_model=True, **TRELLIS2_SIZE, **(TRELLIS2_PROJ if proj else {}),
                     operations=comfy.ops.disable_weight_init, dtype=torch.float32)
    return _random_state_dict(model, seed=3 + proj)


CHECKPOINTS = {
    "hunyuan3d": _hunyuan3d_checkpoint,
    "hunyuan3d21": _hunyuan3d21_checkpoint,
    "trellis2": _trellis2_checkpoint,
    "pixal3d": lambda: _trellis2_checkpoint(proj=True),
}


@contextlib.contextmanager
def _trellis2_size():
    saved = comfy.supported_models.Trellis2.unet_extra_config
    comfy.supported_models.Trellis2.unet_extra_config = {**saved, **TRELLIS2_SIZE}
    try:
        yield
    finally:
        comfy.supported_models.Trellis2.unet_extra_config = saved


class _Group:
    """xFuser's group coordinator, reduced to the all_gather Raylight calls."""

    def __init__(self, group):
        self.group = group

    def all_gather(self, tensor, dim):
        # xFuser hands its input straight to all_gather_into_tensor, which NCCL refuses for a non-contiguous view
        if not tensor.is_contiguous():
            raise ValueError("Tensors must be contiguous")
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, tensor, group=self.group)
        return torch.cat(parts, dim=dim)


@contextlib.contextmanager
def _patched(module, **attrs):
    saved = {name: getattr(module, name) for name in attrs}
    for name, value in attrs.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


@contextlib.contextmanager
def _sequence_parallel(group):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    sp = {"get_sp_group": lambda: _Group(group), "get_sequence_parallel_rank": lambda: rank,
          "get_sequence_parallel_world_size": lambda: world}
    with _patched(diffusion_utils, **sp), _patched(trellis2_usp, **sp):
        yield


@contextlib.contextmanager
def _cfg_parallel(group):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    with _patched(cfg_utils, get_cfg_group=lambda: _Group(group), get_classifier_free_guidance_rank=lambda: rank,
                  get_classifier_free_guidance_world_size=lambda: world):
        yield


def _hunyuan3d_inputs(device, batch, latents=37, context=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(batch, 8, latents, generator=g).to(device), "timestep": torch.rand(batch, generator=g).to(device),
            "context": torch.randn(batch, context, 24, generator=g).to(device),
            "guidance": torch.full((batch,), 5.0).to(device)}


def _hunyuan3d21_inputs(device, items, cfg_copies=1, latents=36, context=9, seed=0):
    # Every cond_or_uncond copy shares its latents, as in sampling: HunYuanDiTPlain's swap of the batch halves only
    # commutes with the model for such a batch.
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(items, 8, latents, generator=g).repeat(cfg_copies, 1, 1).to(device),
            "t": torch.rand(items, generator=g).repeat(cfg_copies).to(device),
            "context": torch.randn(items * cfg_copies, context, 1024, generator=g).to(device),
            "transformer_options": {"cond_or_uncond": list(range(cfg_copies))[::-1]}}


def _trellis2_inputs(device, mode, cfg_copies=1, counts=(23,), context=7, proj=False, seed=0):
    """A Trellis2 call as the stage nodes set it up: one latent per item, coords shared by the cond and uncond copies."""
    g = torch.Generator().manual_seed(seed)
    items = len(counts)
    batch = items * cfg_copies
    inputs = {"timestep": torch.rand(1, generator=g).repeat(batch).to(device),
              "context": torch.randn(batch, context, 32, generator=g).to(device),
              "embeds": torch.randn(batch, context + 2, 32, generator=g).to(device),
              "transformer_options": {"cond_or_uncond": list(range(cfg_copies))}}
    if mode == "structure_generation":
        inputs["x"] = torch.randn(items, 32, 16, 16, 16, generator=g).repeat(cfg_copies, 1, 1, 1, 1).to(device)
        if proj:
            inputs["trellis2_proj_feats"] = torch.randn(items, 16 ** 3, 1024, generator=g).to(device)
        return inputs
    coords = []
    for item, count in enumerate(counts):
        cells = torch.randperm(32 ** 3, generator=g)[:count].sort().values
        coords.append(torch.stack((torch.full_like(cells, item), cells // 1024, cells // 32 % 32, cells % 32), dim=1))
    coords = torch.cat(coords).int()
    x = torch.zeros(items, 32, max(counts), 1)
    for item, count in enumerate(counts):
        x[item, :, :count] = torch.randn(32, count, 1, generator=g)
    inputs.update({"x": x.repeat(cfg_copies, 1, 1, 1).to(device), "trellis2_coords": coords.to(device),
                   "trellis2_coord_counts": torch.tensor(counts).to(device), "trellis2_generation_mode": mode})
    if mode == "texture_generation":
        inputs["trellis2_shape_slat"] = torch.randn(sum(counts), 32, generator=g).to(device)
    elif proj:
        # one row per voxel; Trellis2.forward repeats it for the CFG copies
        inputs["trellis2_proj_feats"] = torch.randn(sum(counts), 2048, generator=g).to(device)
    return inputs


def _forward(patcher, inputs, transformer_options=None):
    inputs = dict(inputs)
    options = {**inputs.pop("transformer_options", {}), **(transformer_options or {})}
    return patcher.model.diffusion_model(**inputs, transformer_options=options)


def _cfg_wrapper_options():
    # Hunyuan3D 2.0 runs the DIFFUSION_MODEL wrappers, so it keeps Raylight's generic CFG wrapper
    wrappers = {comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL: {"raylight_cfg": [cfg_parallel_forward_wrapper]}}
    return {"wrappers": wrappers}


class ThreeDParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            raise unittest.SkipTest("Run with torchrun to exercise distributed FSDP")
        cls.rank, cls.world = dist.get_rank(), dist.get_world_size()
        cls.device = torch.device("cuda", torch.cuda.current_device()) if CUDA else torch.device("cpu")
        cls.mesh = init_device_mesh(cls.device.type, (cls.world,))
        workdir = [tempfile.mkdtemp(prefix="raylight_3d_") if cls.rank == 0 else None]
        dist.broadcast_object_list(workdir, src=0)
        cls.workdir = workdir[0]
        cls.sp_groups = cls.cfg_groups = None
        if cls.world == 4:
            # every rank has to create every group, in the same order
            cls.sp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
            cls.cfg_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if cls.rank == 0:
            shutil.rmtree(cls.workdir, ignore_errors=True)

    def _path(self, name):
        path = os.path.join(self.workdir, f"{name}.safetensors")
        if self.rank == 0 and not os.path.exists(path):
            safetensors.torch.save_file(CHECKPOINTS[name](), path)
        dist.barrier()
        return path

    def _reference(self, name):
        with _trellis2_size():
            reference = comfy.sd.load_diffusion_model(self._path(name), model_options=dict(MODEL_OPTIONS))
        comfy.model_management.load_models_gpu([reference])
        return reference

    def _fsdp(self, name, usp=False, cfg=False):
        with _trellis2_size():
            patcher, state_dict = fsdp_load_diffusion_model(self._path(name), self.rank, self.mesh, False,
                                                            model_options=dict(MODEL_OPTIONS))
        patcher.set_fsdp_state_dict(state_dict)
        patcher.patch_fsdp()
        comfy.model_management.load_models_gpu([patcher])
        # the order the worker registers its ON_LOAD callbacks in: USP, then CFG
        if usp:
            USPInjectRegistry.inject(patcher, self.device, 0, False, True)
        if cfg and CFGParallelInjectRegistry.has_direct_handler(patcher):
            CFGParallelInjectRegistry.inject_direct(patcher)
        return patcher

    def _all_ranks(self, ok):
        flag = torch.tensor([float(ok)], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _assert_close_on_all_ranks(self, actual, expected, rtol=1e-4, atol=1e-4):
        # every rank fails together, so one rank's assertion cannot leave the others waiting in a collective
        same_shape = actual.shape == expected.shape
        ok = same_shape and bool(torch.isfinite(expected).all()) and torch.allclose(actual, expected, rtol=rtol, atol=atol)
        if not self._all_ranks(ok):
            err = (actual - expected).abs().max().item() if same_shape else float("nan")
            self.fail(f"rank {self.rank}: outputs differ on at least one rank (local max abs err {err:.3e}, "
                      f"shapes {tuple(actual.shape)} / {tuple(expected.shape)})")

    def _sp_group(self):
        return self.sp_groups[self.rank // 2] if self.world == 4 else None

    def _cfg_group(self):
        return self.cfg_groups[self.rank % 2] if self.world == 4 else None

    def _check(self, name, inputs, usp=False, cfg=False, transformer_options=None):
        if cfg and self.world not in (2, 4):
            self.skipTest("CFG parallel needs two CFG ranks")
        reference, patcher = self._reference(name), self._fsdp(name, usp=usp, cfg=cfg)
        sp = _sequence_parallel(self._sp_group()) if usp else contextlib.nullcontext()
        guided = _cfg_parallel(self._cfg_group()) if cfg else contextlib.nullcontext()
        with sp, guided:
            actual = _forward(patcher, inputs, transformer_options)
        self._assert_close_on_all_ranks(actual, _forward(reference, inputs))

    # Hunyuan3D 2.0: 37 latents and 11 condition tokens split unevenly over 2, 3 and 4 ranks

    def test_hunyuan3d_fsdp(self):
        self._check("hunyuan3d", _hunyuan3d_inputs(self.device, 2, seed=self.rank))

    def test_hunyuan3d_usp(self):
        self._check("hunyuan3d", _hunyuan3d_inputs(self.device, 2, seed=5), usp=True)

    def test_hunyuan3d_usp_with_block_replace(self):
        def replace(args, extra):
            out = extra["original_block"](args)
            return {"img": out["img"] * 0.75}

        options = {"patches_replace": {"dit": {("single_block", 1): replace}}}
        reference, patcher = self._reference("hunyuan3d"), self._fsdp("hunyuan3d", usp=True)
        inputs = _hunyuan3d_inputs(self.device, 1, seed=6)
        with _sequence_parallel(self._sp_group()):
            actual = _forward(patcher, inputs, dict(options))
        self._assert_close_on_all_ranks(actual, _forward(reference, inputs, dict(options)))

    def test_hunyuan3d_cfg(self):
        if self.world == 4:
            self.skipTest("covered by the USP x CFG test")
        self._check("hunyuan3d", _hunyuan3d_inputs(self.device, 2, seed=7), cfg=True, transformer_options=_cfg_wrapper_options())

    def test_hunyuan3d_usp_and_cfg(self):
        if self.world != 4:
            self.skipTest("USP x CFG needs four ranks")
        self._check("hunyuan3d", _hunyuan3d_inputs(self.device, 2, seed=8), usp=True, cfg=True,
                    transformer_options=_cfg_wrapper_options())

    # Hunyuan3D 2.1: a time token in front of the latents, MoE feed-forwards, non-persistent timestep frequencies

    def test_hunyuan3d21_fsdp(self):
        self._check("hunyuan3d21", _hunyuan3d21_inputs(self.device, 1, cfg_copies=2, seed=self.rank))

    def test_hunyuan3d21_usp(self):
        # each rank routes different tokens through the experts
        self._check("hunyuan3d21", _hunyuan3d21_inputs(self.device, 1, cfg_copies=2, seed=9), usp=True)

    def test_hunyuan3d21_cfg(self):
        self._check("hunyuan3d21", _hunyuan3d21_inputs(self.device, 1, cfg_copies=2, seed=10), usp=self.world == 4, cfg=True)

    def test_hunyuan3d21_cfg_with_two_latents_per_condition(self):
        self._check("hunyuan3d21", _hunyuan3d21_inputs(self.device, 2, cfg_copies=2, seed=11), usp=self.world == 4, cfg=True)

    def test_hunyuan3d21_unguided_call_runs_whole_under_cfg(self):
        self._check("hunyuan3d21", _hunyuan3d21_inputs(self.device, 1, seed=12), usp=self.world == 4, cfg=True)

    # TRELLIS.2: dense 16^3 structure model, sparse shape (512 and 1024) and texture models

    def test_trellis2_fsdp(self):
        for mode in ("structure_generation", "shape_generation_512", "shape_generation", "texture_generation"):
            with self.subTest(mode=mode):
                self._check("trellis2", _trellis2_inputs(self.device, mode, seed=13 + self.rank))

    def test_trellis2_usp(self):
        for mode in ("structure_generation", "shape_generation_512", "shape_generation", "texture_generation"):
            with self.subTest(mode=mode):
                self._check("trellis2", _trellis2_inputs(self.device, mode, cfg_copies=2, seed=14), usp=True)

    def test_trellis2_usp_with_items_of_different_sizes(self):
        for mode in ("shape_generation", "texture_generation"):
            with self.subTest(mode=mode):
                self._check("trellis2", _trellis2_inputs(self.device, mode, cfg_copies=2, counts=(19, 30), seed=15), usp=True)

    def test_trellis2_cfg(self):
        for mode in ("structure_generation", "shape_generation", "texture_generation"):
            with self.subTest(mode=mode):
                self._check("trellis2", _trellis2_inputs(self.device, mode, cfg_copies=2, seed=16), usp=self.world == 4, cfg=True)

    def test_trellis2_cfg_with_two_items(self):
        self._check("trellis2", _trellis2_inputs(self.device, "shape_generation", cfg_copies=2, counts=(19, 30), seed=17),
                    usp=self.world == 4, cfg=True)

    def test_trellis2_unguided_call_runs_whole_under_cfg(self):
        self._check("trellis2", _trellis2_inputs(self.device, "texture_generation", seed=18), usp=self.world == 4, cfg=True)

    def test_pixal3d_usp_and_cfg(self):
        # projected per-voxel / per-cell features are split with the tokens, and zeroed for the uncond copy
        for mode in ("structure_generation", "shape_generation"):
            with self.subTest(mode=mode):
                inputs = _trellis2_inputs(self.device, mode, cfg_copies=2, proj=True, seed=19)
                self._check("pixal3d", inputs, usp=True)
                if self.world in (2, 4):
                    self._check("pixal3d", inputs, usp=self.world == 4, cfg=True)

    def test_trellis2_sampler_with_cfg_override(self):
        if self.world not in (2, 4):
            self.skipTest("CFG parallel needs two CFG ranks")
        reference, patcher = self._reference("trellis2"), self._fsdp("trellis2", usp=self.world == 4, cfg=True)
        inputs = _trellis2_inputs(self.device, "shape_generation", seed=20)

        def sample(model):
            # guided for the first half of the steps, cfg 1.0 (one batch, nothing to split) for the rest
            model = RayCFGOverride.patch.__wrapped__(RayCFGOverride(), model, 1.0, 0.5, 1.0)
            extra = {key: inputs[key] for key in ("trellis2_coords", "trellis2_coord_counts", "trellis2_generation_mode")}
            positive = [[inputs["context"][:1].cpu(), {"embeds": inputs["embeds"][:1].cpu(), **extra}]]
            negative = [[torch.zeros_like(positive[0][0]), {"embeds": torch.zeros_like(positive[0][1]["embeds"]), **extra}]]
            latent = torch.zeros_like(inputs["x"]).cpu()
            noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(21))
            return comfy.sample.sample(model, noise, 4, 6.0, "euler", "simple", positive, negative, latent, disable_pbar=True, seed=21)

        sp = _sequence_parallel(self._sp_group()) if self.world == 4 else contextlib.nullcontext()
        with sp, _cfg_parallel(self._cfg_group()):
            actual = sample(patcher)
        self._assert_close_on_all_ranks(actual, sample(reference))


if __name__ == "__main__":
    dist.init_process_group("nccl" if CUDA else "gloo", timeout=timedelta(seconds=120))
    try:
        unittest.main()
    finally:
        dist.destroy_process_group()
