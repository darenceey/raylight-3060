"""A weight dtype picked in the loader survives Raylight's FSDP loader.

FSDPModelPatcher reports itself dynamic, so the loader assigns the checkpoint tensors instead of copying them into
the model's parameters, and an fp8 weight dtype asked of a bf16 file used to come out as bf16 again.
"""
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import safetensors.torch

import comfy.ops
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

from raylight.comfy_dist.sd import fsdp_load_diffusion_model


def test_fsdp_loader_keeps_the_requested_weight_dtype(tmp_path):
    model = QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=1, attention_head_dim=128,
                                          num_attention_heads=1, context_in_dim=32, mlp_ratio=1,
                                          operations=comfy.ops.disable_weight_init, dtype=torch.bfloat16)
    path = str(tmp_path / "qwen21_bf16.safetensors")
    safetensors.torch.save_file({k: torch.randn(v.shape, dtype=torch.bfloat16) * 0.05 for k, v in model.state_dict().items()}, path)

    _, state_dict = fsdp_load_diffusion_model(path, 0, None, False, model_options={"dtype": torch.float8_e4m3fn})

    assert {str(v.dtype) for v in state_dict.values()} == {"torch.float8_e4m3fn"}
