"""Qwen Image 2.1 through Raylight's FSDP path, checked on every rank against ComfyUI's own unsharded model.

    torchrun --nproc_per_node=2 tests/test_qwen_image21_fsdp.py
    torchrun --nproc_per_node=4 tests/test_qwen_image21_fsdp.py   # adds USP x CFG on top of FSDP

Runs on CUDA with NCCL when there is a card, otherwise on CPU with gloo. Under plain pytest there is no process
group and the tests skip.
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

# torchrun runs this file directly, without conftest: find Raylight's src and the ComfyUI checkout it sits in
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CUDA = torch.cuda.is_available()
if not CUDA:
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import safetensors.torch
from torch.distributed.device_mesh import init_device_mesh

import comfy.ops
import comfy.patcher_extension
import comfy.sample
import comfy.sd
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor

import raylight.diffusion_models.qwen_image.xdit_context_parallel21 as usp
import raylight.diffusion_models.utils as diffusion_utils
import raylight.distributed_modules.cfg_utils as cfg_utils
from raylight.comfy_dist.kitchen_distributed import temporary_sitepkg_ck_patches
from raylight.comfy_dist.model_patcher import FSDPModelPatcher
from raylight.comfy_dist.sd import fsdp_load_diffusion_model
from raylight.diffusion_models.qwen_image.xdit_cfg_parallel import cfg_parallel_forward_wrapper_qwen21

MODEL_OPTIONS = {"dtype": torch.float32}
QUANT_FORMATS = ("float8_e4m3fn", "nvfp4", "mxfp8", "int8_tensorwise")
# What a CPU run cannot shard; ComfyUI's own model is checked first, so formats it cannot run on the device skip too.
CPU_UNSHARDABLE = {
    "float8_e4m3fn": "gloo cannot all-gather float8",
    "mxfp8": "Raylight's MXFP8 sharding swizzles block scales, which CPU does not support",
}


def _checkpoint(quant_format=None):
    torch.manual_seed(0)
    model = QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=2, attention_head_dim=128,
                                          num_attention_heads=2, context_in_dim=64, mlp_ratio=3,
                                          operations=comfy.ops.disable_weight_init, dtype=torch.float32)
    sd = {}
    for name, param in model.state_dict().items():
        weight = torch.randn(param.shape) * 0.05
        if quant_format is None or weight.ndim != 2:
            sd[name] = weight
            continue
        # the layout a ComfyUI-quantized checkpoint stores: weight payload plus a comfy_quant tag per linear
        qt = QuantizedTensor.from_float(weight.to(torch.bfloat16), QUANT_ALGOS[quant_format]["comfy_tensor_layout"])
        sd.update(qt.state_dict(name))
        sd[name[:-len("weight")] + "comfy_quant"] = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)
    return sd


def _inputs(device, batch, h, w, text_len, refs=(), slots=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(batch, 64, h, w, generator=g).to(device),
        "timesteps": torch.rand(batch, generator=g).to(device),
        "context": torch.randn(batch, text_len, 64, generator=g).to(device),
        "ref_latents": [torch.randn(batch, 64, rh, rw, generator=g).to(device) for rh, rw in refs] or None,
        "image_slots": slots,
    }


def _kitchen_patches(patcher):
    # the worker samples an FSDP model inside patch_enable_comfy_kitchen_fsdp, i.e. with these patches installed
    return temporary_sitepkg_ck_patches() if isinstance(patcher, FSDPModelPatcher) else contextlib.nullcontext()


def _forward(patcher, inputs, transformer_options=None):
    with _kitchen_patches(patcher):
        return patcher.model.diffusion_model(inputs["x"], inputs["timesteps"], inputs["context"], ref_latents=inputs["ref_latents"],
                                             image_slots=inputs["image_slots"], transformer_options=transformer_options or {})


class _Group:
    def __init__(self, group):
        self.group = group

    def all_gather(self, tensor, dim):
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(parts, tensor.contiguous(), group=self.group)
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
    with _patched(usp, get_sp_group=lambda: _Group(group), get_sequence_parallel_rank=lambda: rank,
                  get_sequence_parallel_world_size=lambda: world), \
            _patched(diffusion_utils, get_sequence_parallel_world_size=lambda: world):
        yield


@contextlib.contextmanager
def _cfg_parallel(group):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    with _patched(cfg_utils, get_cfg_group=lambda: _Group(group), get_classifier_free_guidance_rank=lambda: rank,
                  get_classifier_free_guidance_world_size=lambda: world):
        yield


def _cfg_options():
    wrappers = {comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL: {"raylight_cfg": [cfg_parallel_forward_wrapper_qwen21]}}
    return {"wrappers": wrappers}


def _inject_usp(patcher):
    diffusion_model = patcher.model.diffusion_model
    diffusion_model._forward = types.MethodType(usp.usp_dit_forward, diffusion_model)


class QwenImage21FSDPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            raise unittest.SkipTest("Run with torchrun to exercise distributed FSDP")
        cls.rank, cls.world = dist.get_rank(), dist.get_world_size()
        cls.device = torch.device("cuda", torch.cuda.current_device()) if CUDA else torch.device("cpu")
        cls.mesh = init_device_mesh(cls.device.type, (cls.world,))
        workdir = [tempfile.mkdtemp(prefix="raylight_qwen21_") if cls.rank == 0 else None]
        dist.broadcast_object_list(workdir, src=0)
        cls.workdir = workdir[0]

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if cls.rank == 0:
            shutil.rmtree(cls.workdir, ignore_errors=True)

    def _path(self, quant_format=None):
        path = os.path.join(self.workdir, f"qwen21_{quant_format or 'fp32'}.safetensors")
        if self.rank == 0 and not os.path.exists(path):
            safetensors.torch.save_file(_checkpoint(quant_format), path)
        dist.barrier()
        return path

    def _reference(self, quant_format=None):
        reference = comfy.sd.load_diffusion_model(self._path(quant_format), model_options=dict(MODEL_OPTIONS))
        reference.model.to(self.device)
        return reference

    def _fsdp(self, quant_format=None):
        patcher, state_dict = fsdp_load_diffusion_model(self._path(quant_format), self.rank, self.mesh, False,
                                                        model_options=dict(MODEL_OPTIONS))
        patcher.set_fsdp_state_dict(state_dict)
        patcher.patch_fsdp()
        # what FSDPModelPatcher.load sets; for Qwen 2.1 it also re-enables the prefix cache unless FSDP keeps it off
        patcher.model.current_patcher = patcher
        return patcher

    def _all_ranks(self, ok):
        flag = torch.tensor([float(ok)], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _assert_close_on_all_ranks(self, actual, expected, rtol=1e-5, atol=1e-5):
        # every rank fails together, so one rank's assertion cannot leave the others waiting in a collective
        same_shape = actual.shape == expected.shape
        if not self._all_ranks(same_shape and torch.allclose(actual, expected, rtol=rtol, atol=atol)):
            err = (actual - expected).abs().max().item() if same_shape else float("nan")
            self.fail(f"rank {self.rank}: outputs differ on at least one rank (local max abs err {err:.3e})")

    def test_dp_ranks_with_different_prompts_match_full_model(self):
        reference, patcher = self._reference(), self._fsdp()
        # rank-dependent text lengths and reference images: each rank still runs every block exactly once
        if self.rank % 2:
            inputs = _inputs(self.device, 1, 3, 5, text_len=7, refs=[(2, 3)], slots=[3], seed=self.rank)
        else:
            inputs = _inputs(self.device, 1, 4, 6, text_len=5, seed=self.rank)
        self._assert_close_on_all_ranks(_forward(patcher, inputs), _forward(reference, inputs))

    def test_sampler_with_rank_local_cache_choices_matches_full_model(self):
        reference, patcher = self._reference(), self._fsdp()

        def sample(model, options):
            model = model.clone()
            if options is not None:
                model.model_options.setdefault("transformer_options", {})["qwen_image21_cache"] = options
            seed = 100 + self.rank
            latent = torch.zeros(1, 64, 4, 6)
            noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed))
            g = torch.Generator().manual_seed(seed + 1)
            positive = [[torch.randn(1, 5, 64, generator=g), {}]]
            negative = [[torch.randn(1, 5, 64, generator=g), {}]]
            with _kitchen_patches(model):
                return comfy.sample.sample(model, noise, 3, 2.5, "euler", "simple", positive, negative, latent,
                                           disable_pbar=True, seed=seed)

        # Even ranks ask for a prefix K/V cache. Odd ranks either turn it off, which used to leave the ranks running
        # different numbers of block forwards, or leave the choice to ComfyUI's dynamic VRAM path, which used to load
        # the cache through a method FSDPModelPatcher does not have.
        for odd_rank_options in ({"device": "off"}, None):
            with self.subTest(odd_rank_options=odd_rank_options):
                options = {"device": "cpu"} if self.rank % 2 == 0 else odd_rank_options
                self._assert_close_on_all_ranks(sample(patcher, options), sample(reference, options), rtol=1e-4, atol=1e-4)

    def test_usp_matches_full_model(self):
        reference, patcher = self._reference(), self._fsdp()
        _inject_usp(patcher)
        # 15 target tokens do not split evenly, and the reference image sits inside the text
        inputs = _inputs(self.device, 1, 3, 5, text_len=6, refs=[(2, 2)], slots=[2], seed=7)
        with _sequence_parallel(None):
            actual = _forward(patcher, inputs)
        self._assert_close_on_all_ranks(actual, _forward(reference, inputs), rtol=1e-4, atol=1e-4)

    def test_usp_with_block_patches_matches_full_model(self):
        reference, patcher = self._reference(), self._fsdp()
        _inject_usp(patcher)
        inputs = _inputs(self.device, 1, 3, 5, text_len=6, refs=[(2, 2)], slots=[2], seed=8)

        def replace(args, extra):
            out = extra["original_block"](args)
            return {"img": out["img"] * 0.75}

        options = {"patches_replace": {"dit": {("single_block", 1): replace}},
                   "patches": {"single_block": [lambda args: {"img": args["img"] + 0.01 * args["block_index"]}]}}
        with _sequence_parallel(None):
            actual = _forward(patcher, inputs, dict(options))
        self._assert_close_on_all_ranks(actual, _forward(reference, inputs, dict(options)), rtol=1e-4, atol=1e-4)

    def test_cfg_matches_full_model(self):
        if self.world != 2:
            self.skipTest("CFG parallel over the whole world needs exactly two ranks")
        reference, patcher = self._reference(), self._fsdp()
        inputs = _inputs(self.device, 2, 4, 6, text_len=5, refs=[(2, 2)], slots=[1], seed=9)
        with _cfg_parallel(None):
            actual = _forward(patcher, inputs, _cfg_options())
        self._assert_close_on_all_ranks(actual, _forward(reference, inputs), rtol=1e-4, atol=1e-4)

    def test_usp_and_cfg_matches_full_model(self):
        if self.world != 4:
            self.skipTest("USP x CFG needs four ranks")
        # every rank has to create every group, in the same order
        sp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        cfg_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        reference, patcher = self._reference(), self._fsdp()
        _inject_usp(patcher)
        inputs = _inputs(self.device, 2, 3, 5, text_len=6, refs=[(2, 2)], slots=[2], seed=10)
        with _sequence_parallel(sp_groups[self.rank // 2]), _cfg_parallel(cfg_groups[self.rank % 2]):
            actual = _forward(patcher, inputs, _cfg_options())
        self._assert_close_on_all_ranks(actual, _forward(reference, inputs), rtol=1e-4, atol=1e-4)

    def test_quantized_checkpoints_match_full_model(self):
        for quant_format in QUANT_FORMATS:
            with self.subTest(quant_format=quant_format):
                if not CUDA and quant_format in CPU_UNSHARDABLE:
                    self.skipTest(CPU_UNSHARDABLE[quant_format])
                inputs = _inputs(self.device, 1, 4, 6, text_len=5, refs=[(2, 2)], slots=[1], seed=11 + self.rank)
                expected, error = None, "see another rank"
                try:
                    expected = _forward(self._reference(quant_format), inputs)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                if not self._all_ranks(expected is not None):
                    self.skipTest(f"ComfyUI itself cannot run {quant_format} on {self.device.type}: {error}")
                self._assert_close_on_all_ranks(_forward(self._fsdp(quant_format), inputs), expected)


if __name__ == "__main__":
    if CUDA:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl" if CUDA else "gloo", timeout=timedelta(seconds=120))
    try:
        unittest.main()
    finally:
        dist.destroy_process_group()
