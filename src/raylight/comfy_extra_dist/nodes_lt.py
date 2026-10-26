import logging
import math
import re

import comfy.model_sampling
import comfy.samplers
import node_helpers
import ray
import torch
import torchaudio

from xfuser.core.distributed import get_classifier_free_guidance_world_size

from .nodes_custom_sampler import _clear_ray_worker_vram_after_sampling, _make_ray_guider
from .ray_patch_decorator import ray_patch, run_model_patch, save_model_state, use_model_state


def _get_cfg_world_size():
    try:
        world_size = int(get_classifier_free_guidance_world_size())
    except Exception:
        return 1
    return 1 if world_size <= 0 else world_size


def _calc_cond_pass(args, cond, model_options):
    """Run one extra model pass on cond alone, as LTXV's post-CFG guidance nodes do.

    Raylight's CFG-parallel forward needs one batch chunk per CFG rank, so under CFG parallel the cond goes once to
    every rank, and every rank computes the same prediction.
    """
    x = args["input"]
    conds = [cond]
    cfg_world_size = _get_cfg_world_size()
    if cfg_world_size > 1 and x.shape[0] != cfg_world_size:
        conds = [list(cond) for _ in range(cfg_world_size)]
    return comfy.samplers.calc_cond_batch(args["model"], conds, x, args["sigma"], model_options)[0]


def _sigma_range(model, start_percent, end_percent):
    model_sampling = model.get_model_object("model_sampling")
    return model_sampling.percent_to_sigma(start_percent), model_sampling.percent_to_sigma(end_percent)


def _transformer_options_pass(args, **options):
    model_options = args["model_options"].copy()
    model_options["transformer_options"] = {**model_options.get("transformer_options", {}), **options}
    return model_options


def _patch_ltx_reference_audio_guidance(model, identity_guidance_scale, start_percent, end_percent):
    m = model.clone()
    scale = identity_guidance_scale
    model_sampling = m.get_model_object("model_sampling")
    sigma_start = model_sampling.percent_to_sigma(start_percent)
    sigma_end = model_sampling.percent_to_sigma(end_percent)

    def post_cfg_function(args):
        if scale == 0:
            return args["denoised"]

        sigma = args["sigma"]
        sigma_ = sigma[0].item()
        if sigma_ > sigma_start or sigma_ < sigma_end:
            return args["denoised"]

        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        cfg_result = args["denoised"]

        # Strip ref_audio from conditioning for the no-reference pass.
        noref_cond = []
        for entry in cond:
            new_entry = entry.copy()
            mc = new_entry.get("model_conds", {}).copy()
            mc.pop("ref_audio", None)
            new_entry["model_conds"] = mc
            noref_cond.append(new_entry)

        pred_noref = _calc_cond_pass(args, noref_cond, args["model_options"].copy())

        return cfg_result + (cond_pred - pred_noref) * scale

    m.set_model_sampler_post_cfg_function(post_cfg_function)
    return m


class RayModelSamplingLTXV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(self, model, max_shift, base_shift, latent=None):
        m = model.clone()

        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = tokens * mm + b

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return m


class RayLTXVReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference_audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "identity_guidance_scale": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01},
                ),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("ray_actors", "positive", "negative")
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    def patch(
        self,
        ray_actors,
        positive,
        negative,
        reference_audio,
        audio_vae,
        identity_guidance_scale,
        start_percent,
        end_percent,
    ):
        sample_rate = reference_audio["sample_rate"]
        vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
        if vae_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(reference_audio["waveform"], sample_rate, vae_sample_rate)
        else:
            waveform = reference_audio["waveform"]

        audio_latents = audio_vae.encode(waveform.movedim(1, -1))
        b, c, t, f = audio_latents.shape
        ref_tokens = audio_latents.permute(0, 2, 1, 3).reshape(b, t, c * f)
        ref_audio = {"tokens": ref_tokens}

        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})

        use_model_state(ray_actors)
        replay = run_model_patch(
            ray_actors, _patch_ltx_reference_audio_guidance, identity_guidance_scale, start_percent, end_percent
        )

        return (save_model_state(ray_actors, replay=replay), positive, negative)


class RayLTXVSpatioTemporalGuidance:
    """LTXVSpatioTemporalGuidance: one extra pass per step with the self-attention of the given blocks reduced to its
    values, and guidance away from it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "blocks": ("STRING", {"default": "29", "tooltip": "Comma-separated transformer block indices to perturb."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, scale, blocks, start_percent, end_percent):
        block_set = frozenset(int(b) for b in re.findall(r"\d+", blocks))
        m = model.clone()
        sigma_start, sigma_end = _sigma_range(m, start_percent, end_percent)

        def post_cfg_function(args):
            if scale == 0 or not block_set:
                return args["denoised"]
            sigma_ = args["sigma"][0].item()
            if sigma_ > sigma_start or sigma_ < sigma_end:
                return args["denoised"]
            perturbed = _calc_cond_pass(args, args["cond"], _transformer_options_pass(args, stg_self_attn_blocks=block_set))
            return args["denoised"] + (args["cond_denoised"] - perturbed) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return m


class RayLTXVModalityGuidance:
    """LTXVModalityGuidance: one extra pass per step with the audio-video cross-attention cut in both directions, and
    guidance toward the coupled prediction (audio-visual sync)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "modality_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra/guidance"

    @ray_patch
    def patch(self, model, modality_scale, start_percent, end_percent):
        m = model.clone()
        sigma_start, sigma_end = _sigma_range(m, start_percent, end_percent)

        def post_cfg_function(args):
            if math.isclose(modality_scale, 1.0):
                return args["denoised"]
            sigma_ = args["sigma"][0].item()
            if sigma_ > sigma_start or sigma_ < sigma_end:
                return args["denoised"]
            uncoupled = _calc_cond_pass(args, args["cond"], _transformer_options_pass(args, a2v_cross_attn=False, v2a_cross_attn=False))
            return args["denoised"] + (args["cond_denoised"] - uncoupled) * (modality_scale - 1.0)

        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return m


class RayLTXVDualCFGGuider:
    """LTXVDualCFGGuider: separate CFG scales for the video and audio parts of an LTXV-AV latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "Raylight/extra/custom_sampling/guiders"

    def get_guider(self, ray_actors, positive, negative, video_cfg, audio_cfg):
        return (_make_ray_guider(ray_actors, "ltxav_dual_cfg", positive=positive, negative=negative,
                                 video_cfg=video_cfg, audio_cfg=audio_cfg),)


def predict_ltxv_duration(duration_head, processed, video_dim, frame_rate, min_seconds, max_seconds):
    """LTXVDurationPredictor's head on caption tokens a worker made; returns (num_frames, seconds)."""
    import comfy.ldm.lightricks.duration_head as duration
    import comfy.model_management

    head = duration_head.model
    if not isinstance(head, duration.DurationHead):
        raise ValueError("The connected model_patch is not an LTX duration head.")
    comfy.model_management.load_models_gpu([duration_head])
    device = duration_head.load_device
    with torch.no_grad():
        seconds = float(head(processed[..., :video_dim].to(device), processed[..., video_dim:].to(device))[0])
    return duration.seconds_to_num_frames(seconds, frame_rate, min_seconds, max_seconds), seconds


class RayLTXVDurationPredictor:
    """LTXVDurationPredictor: the natural shot length of a prompt from the LTX duration head. The workers' model runs
    its text connectors on the prompt; the head (loaded with Load Model Patch) runs here."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "positive": ("CONDITIONING",),
                "duration_head": ("MODEL_PATCH", {"tooltip": "LTX duration head loaded with Load Model Patch."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "min_seconds": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 120.0, "step": 0.1}),
                "max_seconds": ("FLOAT", {"default": 20.0, "min": 0.5, "max": 120.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("num_frames", "seconds")
    FUNCTION = "predict"
    CATEGORY = "Raylight/extra"

    def predict(self, ray_actors, positive, duration_head, frame_rate, min_seconds, max_seconds):
        context = positive[0][0][:1]
        unprocessed = positive[0][1].get("unprocessed_ltxav_embeds", False)
        use_model_state(ray_actors)
        # every worker runs the connectors: under FSDP they are sharded across all of them
        processed, video_dim = ray.get([actor.ltxv_caption_tokens.remote(context, unprocessed)
                                        for actor in ray_actors["workers"]])[0]
        # like a Ray sampler: with clear_vram_after_sampling the workers give the VRAM back (GPU 0 is shared with the
        # VAE and text encoders here)
        _clear_ray_worker_vram_after_sampling(ray_actors)
        num_frames, seconds = predict_ltxv_duration(duration_head, processed, video_dim, frame_rate, min_seconds, max_seconds)
        logging.info("[Raylight] LTXV duration head predicted %.2fs -> %d frames @ %.2f fps", seconds, num_frames, frame_rate)
        return (num_frames, seconds)


NODE_CLASS_MAPPINGS = {
    "RayModelSamplingLTXV": RayModelSamplingLTXV,
    "RayLTXVReferenceAudio": RayLTXVReferenceAudio,
    "RayLTXVSpatioTemporalGuidance": RayLTXVSpatioTemporalGuidance,
    "RayLTXVModalityGuidance": RayLTXVModalityGuidance,
    "RayLTXVDualCFGGuider": RayLTXVDualCFGGuider,
    "RayLTXVDurationPredictor": RayLTXVDurationPredictor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayModelSamplingLTXV": "Model Sampling LTXV (Ray)",
    "RayLTXVReferenceAudio": "LTXV Reference Audio (Ray)",
    "RayLTXVSpatioTemporalGuidance": "LTXV Spatio-Temporal Guidance (Ray)",
    "RayLTXVModalityGuidance": "LTXV Modality Guidance (Ray)",
    "RayLTXVDualCFGGuider": "LTXV Dual CFG Guider (Ray)",
    "RayLTXVDurationPredictor": "LTXV Duration Predictor (Ray)",
}
