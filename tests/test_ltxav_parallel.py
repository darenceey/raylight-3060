"""LTX-2.3 and LTX-2.5 (LTXAV) sampling through Raylight's FSDP, USP and CFG paths, checked on every rank against
ComfyUI's own unsharded model, with the LTXV guidance nodes Raylight ships for them.

    torchrun --nproc_per_node=2 tests/test_ltxav_parallel.py
    torchrun --nproc_per_node=3 tests/test_ltxav_parallel.py   # odd world: every sequence needs padding
    torchrun --nproc_per_node=4 tests/test_ltxav_parallel.py   # adds USP x CFG

Runs on CUDA with NCCL when there is a card, otherwise on CPU with gloo. Under plain pytest there is no process
group and the tests skip.

xFuser's Ulysses attention is replaced by its exact equivalent here - each rank's queries against the keys and values
of every rank, padding included - so the test needs no xFuser process groups; the GPU runs on real checkpoints cover
xFuser itself.
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

import json

import safetensors.torch
from torch.distributed.device_mesh import init_device_mesh

import comfy.ldm.modules.attention
import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.sd
import node_helpers
from comfy.ldm.lightricks.av_model import LTXAVModel
from comfy.nested_tensor import NestedTensor

import raylight.distributed_modules.attention as xfuser_attn

_SP_GROUP = {}


def _ulysses_attention(q, k, v, heads, attn_precision=None, **kwargs):
    """What xFuser's Ulysses attention computes: local queries against the whole (padded) key/value sequence."""
    group = _SP_GROUP["group"]
    k, v = group.all_gather(k.contiguous(), dim=1), group.all_gather(v.contiguous(), dim=1)
    return comfy.ldm.modules.attention.optimized_attention(q, k, v, heads, attn_precision=attn_precision)


xfuser_attn.set_attn_type("TORCH_EFFICIENT")
xfuser_attn.set_sync_ulysses(False)
xfuser_attn.make_xfuser_attention = lambda *args: _ulysses_attention

import raylight.comfy_extra_dist.nodes_lt as nodes_lt  # noqa: E402
import raylight.diffusion_models.lightricks.xdit_context_parallel as ltx_usp  # noqa: E402
import raylight.diffusion_models.utils as diffusion_utils  # noqa: E402
import raylight.distributed_modules.cfg_utils as cfg_utils  # noqa: E402
from raylight.comfy_dist.sd import fsdp_load_diffusion_model  # noqa: E402
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry  # noqa: E402
from raylight.distributed_modules.usp import USPInjectRegistry  # noqa: E402
from raylight.distributed_worker.ray_worker import _build_ray_guider, ltxv_caption_tokens  # noqa: E402

MODEL_OPTIONS = {"dtype": torch.float32}
# LTX-2.3's layout (cross-attention AdaLN, gated attention, connectors behind the caption projection) at width 64
LTX23 = {
    "num_attention_heads": 4, "attention_head_dim": 16, "cross_attention_dim": 64,
    "audio_num_attention_heads": 4, "audio_attention_head_dim": 8, "audio_cross_attention_dim": 32,
    "num_layers": 3, "caption_channels": 48, "cross_attention_adaln": True, "caption_proj_before_connector": True,
    "caption_projection_first_linear": False, "apply_gated_attention": True, "connector_apply_gated_attention": True,
    "connector_num_attention_heads": 4, "connector_attention_head_dim": 16, "audio_connector_num_attention_heads": 4,
    "audio_connector_attention_head_dim": 8, "connector_num_layers": 1, "connector_num_learnable_registers": 8,
    "use_middle_indices_grid": True, "causal_temporal_positioning": True, "rope_type": "split",
    "frequencies_precision": "float64", "av_ca_timestep_scale_multiplier": 1000.0, "timestep_scale_multiplier": 1000,
}
# LTX-2.5: no video feed-forward biases, and a learned marker on the tokens of single-pixel-frame latents
LTX25 = {**LTX23, "ff_bias": False}
CONFIGS = {"ltx23": (LTX23, False), "ltx25": (LTX25, True)}


def _checkpoint(name):
    config, keyframes = CONFIGS[name]
    model = LTXAVModel(**config, use_keyframes_abs_pos_embedding=keyframes, operations=comfy.ops.disable_weight_init,
                       dtype=torch.float32)
    g = torch.Generator().manual_seed(len(name) + keyframes)
    state_dict = {"model.diffusion_model." + key: torch.randn(value.shape, generator=g) * 0.05
                  for key, value in model.state_dict().items()}
    return state_dict, {"config": json.dumps({"transformer": config})}


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
    _SP_GROUP["group"] = _Group(group)
    with _patched(diffusion_utils, **sp), _patched(ltx_usp, **sp):
        yield


@contextlib.contextmanager
def _cfg_parallel(group):
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    with _patched(cfg_utils, get_cfg_group=lambda: _Group(group), get_classifier_free_guidance_rank=lambda: rank,
                  get_classifier_free_guidance_world_size=lambda: world), \
            _patched(nodes_lt, get_classifier_free_guidance_world_size=lambda: world):
        yield


def _inputs(frames=3, height=3, width=5, audio=13, context=9, ref_audio=0, first_frame=False, seed=0):
    """An AV latent and its conditioning; the defaults give odd token counts for the video, audio and text.

    first_frame keeps the first latent frame, as LTXVImgToVideoInplace does for image to video: its tokens then get
    their own timestep.
    """
    g = torch.Generator().manual_seed(seed)
    video = torch.randn(1, 128, frames, height, width, generator=g) if first_frame else torch.zeros(1, 128, frames, height, width)
    latent = NestedTensor((video, torch.zeros(1, 8, audio, 16)))
    noise = NestedTensor((torch.randn(1, 128, frames, height, width, generator=g), torch.randn(1, 8, audio, 16, generator=g)))
    positive = [[torch.randn(1, context, 96, generator=g), {"frame_rate": 25}]]
    negative = [[torch.randn(1, context, 96, generator=g), {"frame_rate": 25}]]
    if ref_audio:
        ref = {"tokens": torch.randn(1, ref_audio, 128, generator=g)}
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref})
    inputs = {"latent": latent, "noise": noise, "positive": positive, "negative": negative}
    if first_frame:
        mask = torch.ones(1, 1, frames, height, width)
        mask[:, :, 0] = 0.0
        inputs["noise_mask"] = NestedTensor((mask, torch.ones(1, 1, audio, 16)))
    return inputs


def _with_latent_guide(inputs, strength, seed=0):
    """LTXVAddLatentGuide on the video latent: a guide frame appended at latent frame 0, attended at strength."""
    from types import SimpleNamespace

    from comfy_extras.nodes_lt import LTXVAddLatentGuide

    g = torch.Generator().manual_seed(seed)
    video, audio = inputs["latent"].unbind()
    guide = {"samples": torch.randn(1, 128, 1, *video.shape[3:], generator=g)}
    vae = SimpleNamespace(downscale_index_formula=(8, 32, 32))
    positive, negative, latent = LTXVAddLatentGuide.execute(inputs["positive"], inputs["negative"], vae, {"samples": video},
                                                            guide, 0, strength).args
    video = latent["samples"]
    noise_video = torch.randn(video.shape, generator=g)
    mask = latent["noise_mask"].expand(-1, -1, -1, *video.shape[3:])
    return {"latent": NestedTensor((video, audio)), "noise": NestedTensor((noise_video, inputs["noise"].unbind()[1])),
            "positive": positive, "negative": negative,
            "noise_mask": NestedTensor((mask, torch.ones(1, 1, audio.shape[2], 16)))}


def _duration_head(device):
    import comfy.model_patcher
    from comfy.ldm.lightricks.duration_head import DurationHead

    head = DurationHead(video_cross_attention_dim=64, audio_cross_attention_dim=32, pooler_hidden_dim=32, mlp_hidden=32)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for param in head.parameters():
            param.copy_(torch.randn(param.shape, generator=g) * 0.1)
    return comfy.model_patcher.ModelPatcher(head, load_device=device, offload_device=torch.device("cpu"))


def _sample(model, inputs, cfg=4.0, steps=2, guider=None):
    if guider is not None:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps).to(model.load_device)
        out = guider.sample(inputs["noise"], inputs["latent"], comfy.samplers.sampler_object("euler"), sigmas, seed=1)
    else:
        out = comfy.sample.sample(model, inputs["noise"], steps, cfg, "euler", "simple", inputs["positive"], inputs["negative"],
                                  inputs["latent"], noise_mask=inputs.get("noise_mask"), disable_pbar=True, seed=1)
    return torch.cat([part.flatten() for part in out.unbind()]) if isinstance(out, NestedTensor) else out.flatten()


class LTXAVParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_initialized():
            raise unittest.SkipTest("Run with torchrun to exercise distributed FSDP")
        cls.rank, cls.world = dist.get_rank(), dist.get_world_size()
        cls.device = torch.device("cuda", torch.cuda.current_device()) if CUDA else torch.device("cpu")
        cls.mesh = init_device_mesh(cls.device.type, (cls.world,))
        workdir = [tempfile.mkdtemp(prefix="raylight_ltxav_") if cls.rank == 0 else None]
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
            state_dict, metadata = _checkpoint(name)
            safetensors.torch.save_file(state_dict, path, metadata=metadata)
        dist.barrier()
        return path

    def _reference(self, name):
        reference = comfy.sd.load_diffusion_model(self._path(name), model_options=dict(MODEL_OPTIONS))
        comfy.model_management.load_models_gpu([reference])
        return reference

    def _fsdp(self, name, usp=False, cfg=False):
        patcher, state_dict = fsdp_load_diffusion_model(self._path(name), self.rank, self.mesh, False,
                                                        model_options=dict(MODEL_OPTIONS))
        patcher.set_fsdp_state_dict(state_dict)
        patcher.patch_fsdp()
        comfy.model_management.load_models_gpu([patcher])
        # what the worker registers: USP on load, the CFG split as a DIFFUSION_MODEL wrapper
        if usp:
            USPInjectRegistry.inject(patcher, self.device, 0, False, True)
        if cfg:
            patcher.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, CFGParallelInjectRegistry.inject(patcher))
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

    def _check(self, name, inputs, usp=False, cfg=False, patch=None, guider=None, sample_cfg=4.0):
        if cfg and self.world not in (2, 4):
            self.skipTest("CFG parallel needs two CFG ranks")
        reference, patcher = self._reference(name), self._fsdp(name, usp=usp, cfg=cfg)
        if patch is not None:
            reference, patcher = patch(reference), patch(patcher)
        sp = _sequence_parallel(self._sp_group()) if usp else contextlib.nullcontext()
        guided = _cfg_parallel(self._cfg_group()) if cfg else contextlib.nullcontext()
        with sp, guided:
            actual = _sample(patcher, inputs, cfg=sample_cfg, guider=guider(patcher) if guider else None)
        self._assert_close_on_all_ranks(actual, _sample(reference, inputs, cfg=sample_cfg, guider=guider(reference) if guider else None))

    def _modes(self):
        """(usp, cfg) combinations this world size runs."""
        modes = [(True, False)]
        if self.world in (2, 4):
            modes.append((self.world == 4, True))
        return modes

    def test_fsdp(self):
        for name in CONFIGS:
            with self.subTest(name=name):
                self._check(name, _inputs(seed=self.rank))

    def test_usp_and_cfg(self):
        for name in CONFIGS:
            for usp, cfg in self._modes():
                with self.subTest(name=name, usp=usp, cfg=cfg):
                    self._check(name, _inputs(seed=1), usp=usp, cfg=cfg)

    def test_image_to_video_first_frame(self):
        for usp, cfg in self._modes():
            with self.subTest(usp=usp, cfg=cfg):
                self._check("ltx25", _inputs(first_frame=True, seed=7), usp=usp, cfg=cfg)

    def test_guide_attention_strength(self):
        # a guide at strength 0.5 weighs the noisy <-> guide attention; 2 frames + the guide give 45 video tokens
        inputs = _with_latent_guide(_inputs(frames=2, seed=8), 0.5, seed=9)
        for usp, cfg in self._modes():
            with self.subTest(usp=usp, cfg=cfg):
                self._check("ltx25", inputs, usp=usp, cfg=cfg)

    def test_duration_predictor(self):
        from comfy_extras.nodes_lt import LTXVDurationPredictor

        reference, patcher = self._reference("ltx25"), self._fsdp("ltx25")
        head = _duration_head(self.device)
        g = torch.Generator().manual_seed(12)
        for unprocessed in (False, True):
            with self.subTest(unprocessed=unprocessed):
                positive = [[torch.randn(1, 9, 96, generator=g), {"unprocessed_ltxav_embeds": unprocessed}]]
                expected = LTXVDurationPredictor.execute(reference, positive, head, 24.0, 0.5, 120.0).args
                # every rank runs the connectors, as the Ray workers do: they are FSDP units
                processed, video_dim = ltxv_caption_tokens(patcher, positive[0][0], unprocessed)
                actual = nodes_lt.predict_ltxv_duration(head, processed, video_dim, 24.0, 0.5, 120.0)
                ok = actual[0] == expected[0] and abs(actual[1] - expected[1]) <= 1e-4 * abs(expected[1])
                if not self._all_ranks(ok):
                    self.fail(f"rank {self.rank}: predicted {actual}, ComfyUI {expected}")

    def test_usp_with_lengths_every_world_divides(self):
        # 12 video, audio and text tokens: no padding at 2, 3 or 4 ranks
        self._check("ltx25", _inputs(frames=2, height=2, width=3, audio=12, context=12, seed=2), usp=True)

    def test_reference_audio(self):
        for usp, cfg in self._modes():
            with self.subTest(usp=usp, cfg=cfg):
                patch = lambda model: nodes_lt._patch_ltx_reference_audio_guidance(model, 3.0, 0.0, 1.0)
                self._check("ltx23", _inputs(ref_audio=6, seed=3), usp=usp, cfg=cfg, patch=patch)

    def test_spatio_temporal_guidance(self):
        node = nodes_lt.RayLTXVSpatioTemporalGuidance()
        patch = lambda model: nodes_lt.RayLTXVSpatioTemporalGuidance.patch.__wrapped__(node, model, 1.5, "1, 2", 0.0, 1.0)
        for usp, cfg in self._modes():
            with self.subTest(usp=usp, cfg=cfg):
                self._check("ltx25", _inputs(seed=4), usp=usp, cfg=cfg, patch=patch)

    def test_modality_guidance(self):
        node = nodes_lt.RayLTXVModalityGuidance()
        patch = lambda model: nodes_lt.RayLTXVModalityGuidance.patch.__wrapped__(node, model, 3.0, 0.0, 1.0)
        for usp, cfg in self._modes():
            with self.subTest(usp=usp, cfg=cfg):
                self._check("ltx25", _inputs(seed=5), usp=usp, cfg=cfg, patch=patch)

    def test_dual_cfg_guider(self):
        inputs = _inputs(seed=6)
        node = nodes_lt.RayLTXVDualCFGGuider()
        (spec,) = node.get_guider({"workers": []}, inputs["positive"], inputs["negative"], 3.0, 7.0)
        for usp, cfg in self._modes():
            with self.subTest(usp=usp, cfg=cfg):
                self._check("ltx25", inputs, usp=usp, cfg=cfg, guider=lambda model: _build_ray_guider(model, spec))


if __name__ == "__main__":
    dist.init_process_group("nccl" if CUDA else "gloo", timeout=timedelta(seconds=120))
    try:
        unittest.main()
    finally:
        dist.destroy_process_group()
