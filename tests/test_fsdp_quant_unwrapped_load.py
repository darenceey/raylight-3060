"""A quantized ComfyUI linear that FSDP leaves unwrapped keeps its weight through Raylight's loader.

patch_fsdp keeps ControlNet-shared modules (img_in, txt_in, time_text_embed, ...) out of FSDP. Such a module's
state_dict carries its quant payload beside the weight, and the loader used to hand that payload back to
load_state_dict, which made the op decode its finished weight again: INT8 came out wrong, FP8 failed to run.
"""
import json

import pytest
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import comfy.ops
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor

from raylight.comfy_dist.fsdp_utils import load_from_full_model_state_dict


@pytest.mark.parametrize("quant_format", ["float8_e4m3fn", "int8_tensorwise"])
def test_unwrapped_quantized_linear_matches_comfy_after_loading(quant_format):
    torch.manual_seed(0)
    qt = QuantizedTensor.from_float(torch.randn(128, 64) * 0.05, QUANT_ALGOS[quant_format]["comfy_tensor_layout"])
    checkpoint = dict(qt.state_dict("weight"))
    checkpoint["comfy_quant"] = torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8)

    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.float32)
    reference = ops.Linear(64, 128, bias=False)
    reference.load_state_dict(dict(checkpoint), strict=False, assign=True)

    # what patch_fsdp gives the loader for a module it did not wrap: the op's own state_dict, then the op on meta
    model = ops.Linear(64, 128, bias=False)
    model.load_state_dict(dict(checkpoint), strict=False, assign=True)
    full_sd = model.state_dict()
    assert "comfy_quant" in full_sd
    model.to("meta")
    load_from_full_model_state_dict(model, full_sd, torch.device("cpu"))

    x = torch.randn(3, 64)
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
