from __future__ import annotations

import inspect
import types

import torch
from comfy.nested_tensor import NestedTensor
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


_DEFAULT_SKIP_EXTRA_KWARGS = frozenset(
    {
        "transformer_options",
        "disable_time_r",
        "data_type",
    }
)


def _find_batch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, NestedTensor):
        for item in value.unbind():
            tensor = _find_batch_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _find_batch_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _find_batch_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _has_cfg_batch(value, cfg_world_size):
    tensor = _find_batch_tensor(value)
    return tensor is not None and tensor.ndim > 0 and tensor.shape[0] == cfg_world_size


def _chunk_tensor(value, cfg_world_size, cfg_rank):
    if value.ndim == 0:
        return value
    if value.shape[0] == cfg_world_size:
        return torch.chunk(value, cfg_world_size, dim=0)[cfg_rank]
    if value.shape[0] == 1:
        return value
    return value


def _chunk_value(value, cfg_world_size, cfg_rank):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _chunk_tensor(value, cfg_world_size, cfg_rank)
    if isinstance(value, NestedTensor):
        return NestedTensor(_chunk_value(item, cfg_world_size, cfg_rank) for item in value.unbind())
    if isinstance(value, dict):
        return {key: _chunk_value(item, cfg_world_size, cfg_rank) for key, item in value.items()}
    if isinstance(value, list):
        return [_chunk_value(item, cfg_world_size, cfg_rank) for item in value]
    if isinstance(value, tuple):
        return tuple(_chunk_value(item, cfg_world_size, cfg_rank) for item in value)
    return value


def _resolve_argument_name(bound_arguments, name):
    if isinstance(name, str):
        return name if name in bound_arguments else None
    for candidate in name:
        if candidate in bound_arguments:
            return candidate
    return None


def _validate_cfg_batch(value, cfg_world_size):
    tensor = _find_batch_tensor(value)
    if tensor is None or tensor.ndim == 0 or tensor.shape[0] != cfg_world_size:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")


def _chunk_extra_kwargs(extra_kwargs, cfg_world_size, cfg_rank, skip_names):
    chunked = dict(extra_kwargs)
    for key, value in extra_kwargs.items():
        if key in skip_names:
            continue
        if _has_cfg_batch(value, cfg_world_size):
            chunked[key] = _chunk_value(value, cfg_world_size, cfg_rank)
    return chunked


def _gather_value(value):
    # all_gather_into_tensor rejects views, and a model output can be one (Qwen Image 2.1 returns a transposed reshape)
    if isinstance(value, torch.Tensor):
        return get_cfg_group().all_gather(value.contiguous(), dim=0)
    if isinstance(value, NestedTensor):
        return NestedTensor(get_cfg_group().all_gather(item.contiguous(), dim=0) for item in value.unbind())
    if isinstance(value, dict):
        return {key: _gather_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_gather_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_gather_value(item) for item in value)
    return value


def cfg_parallel_forward(
    executor,
    *args,
    chunk_names=(),
    validate_name="x",
    auto_chunk_extra_kwargs=True,
    skip_extra_kwargs=(),
    **kwargs,
):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    signature = inspect.signature(executor.original)
    bound = signature.bind_partial(*args, **kwargs)

    validate_arg_name = _resolve_argument_name(bound.arguments, validate_name)
    if validate_arg_name is None:
        raise RuntimeError(f"CFG wrapper could not resolve validation argument {validate_name!r}")
    _validate_cfg_batch(bound.arguments[validate_arg_name], cfg_world_size)

    for name in chunk_names:
        arg_name = _resolve_argument_name(bound.arguments, name)
        if arg_name is None:
            continue
        bound.arguments[arg_name] = _chunk_value(bound.arguments[arg_name], cfg_world_size, cfg_rank)

    if "control" in bound.arguments:
        bound.arguments["control"] = _chunk_value(bound.arguments["control"], cfg_world_size, cfg_rank)

    if auto_chunk_extra_kwargs:
        var_keyword_name = next(
            (param.name for param in signature.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD),
            None,
        )
        if var_keyword_name is not None and var_keyword_name in bound.arguments:
            skip_names = _DEFAULT_SKIP_EXTRA_KWARGS.union(skip_extra_kwargs)
            bound.arguments[var_keyword_name] = _chunk_extra_kwargs(
                bound.arguments[var_keyword_name],
                cfg_world_size,
                cfg_rank,
                skip_names,
            )

    result = executor(*bound.args, **bound.kwargs)
    return _gather_value(result)


# Direct CFG handlers, for models whose forward does not run the DIFFUSION_MODEL wrappers


def direct_cfg_split(transformer_options):
    """Return (cfg_rank, cfg_world_size, rank's transformer_options) when this call holds one cond/uncond chunk per CFG rank.

    Returns None otherwise: cfg 1.0, a CFG Override range, or ComfyUI running cond and uncond as separate calls. Such a
    call runs whole on every CFG rank instead of raising, so one workflow can mix guided and unguided samplers on the
    same workers.
    """
    cfg_world_size = get_classifier_free_guidance_world_size()
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if cond_or_uncond is None or len(cond_or_uncond) != cfg_world_size:
        return None
    cfg_rank = get_classifier_free_guidance_rank()
    options = dict(transformer_options)
    for key in ("cond_or_uncond", "uuids"):
        value = options.get(key)
        if isinstance(value, list) and len(value) == cfg_world_size:
            options[key] = value[cfg_rank:cfg_rank + 1]
    return cfg_rank, cfg_world_size, options


def chunk_cfg_batch(value, cfg_world_size, cfg_rank):
    # ComfyUI stacks one equally sized chunk per cond_or_uncond entry, so the rank's part is a plain chunk of the batch
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    return torch.chunk(value, cfg_world_size, dim=0)[cfg_rank]


def gather_cfg_batch(value):
    return get_cfg_group().all_gather(value.contiguous(), dim=0)


def patch_direct_cfg_forward(model, cfg_forward):
    if getattr(model.forward, "__func__", None) is cfg_forward:
        return
    model._raylight_cfg_original_forward = model.forward
    model.forward = types.MethodType(cfg_forward, model)
