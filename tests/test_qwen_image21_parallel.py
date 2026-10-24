from types import SimpleNamespace

import torch

import raylight.diffusion_models.qwen_image.xdit_context_parallel21 as usp
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy.ldm.qwen_image21.model import block_causal_attention


def _containers(*tensors):
    return tuple(AttentionTensorContainer(t) for t in tensors)


class _Block:
    def __init__(self):
        self.attn = SimpleNamespace(comfy_attention=SimpleNamespace(function=None))

    def __call__(self, states, mod, pe, attn_fn, prefix_len, options):
        qkv = states.unsqueeze(2)
        return states + attn_fn(*_containers(qkv, qkv, qkv), 1)


class _Model:
    out_channels = 4

    def __init__(self, prefix, segments, num_blocks):
        self.prefix = prefix
        self.segments = segments
        self.transformer_blocks = [_Block() for _ in range(num_blocks)]
        self.modulation = lambda temb: torch.zeros(temb.shape[0], 16)
        self.time_text_embed = lambda timesteps, dtype: torch.zeros(timesteps.shape[0], 4)
        self.norm_out = lambda states, temb: states
        self.proj_out = lambda states: states

    def build_sequence(self, x, context, refs, slots):
        states = torch.cat((self.prefix, x.flatten(2).transpose(1, 2)), dim=1)
        return states, torch.empty(1, states.shape[1], 1, 2, 2, 2), self.segments


def _pad_to_two(tensor, dim):
    if tensor.shape[dim] % 2:
        tensor = torch.cat((tensor, torch.zeros_like(tensor.narrow(dim, 0, 1))), dim=dim)
    return tensor, tensor.shape[dim]


class _TwoRankPatch:
    """Run the USP forward as one rank of two, with all_gather answered by the caller."""

    def __init__(self, rank, all_gather):
        self.rank = rank
        self.all_gather = all_gather
        self.saved = {}

    def __enter__(self):
        for name in ("get_sp_group", "get_sequence_parallel_rank", "get_sequence_parallel_world_size", "pad_to_world_size"):
            self.saved[name] = getattr(usp, name)
        self.saved["make"] = usp.comfy.model_prefetch.make_prefetch_queue
        self.saved["pop"] = usp.comfy.model_prefetch.prefetch_queue_pop
        usp.get_sp_group = lambda: SimpleNamespace(all_gather=self.all_gather)
        usp.get_sequence_parallel_rank = lambda: self.rank
        usp.get_sequence_parallel_world_size = lambda: 2
        usp.pad_to_world_size = _pad_to_two
        usp.comfy.model_prefetch.make_prefetch_queue = lambda *args: None
        usp.comfy.model_prefetch.prefetch_queue_pop = lambda *args: None
        return self

    def __exit__(self, *args):
        usp.comfy.model_prefetch.make_prefetch_queue = self.saved.pop("make")
        usp.comfy.model_prefetch.prefetch_queue_pop = self.saved.pop("pop")
        for name, value in self.saved.items():
            setattr(usp, name, value)


def test_qwen21_sharded_attention_matches_block_causal_attention():
    torch.manual_seed(0)
    segments = [(0, 2, torch.ones(2, 2, dtype=torch.bool).tril()), (2, 4, None), (4, 9, None)]
    q, k, v = (torch.randn(1, 9, 2, 4) for _ in range(3))
    expected = block_causal_attention(segments)(*_containers(q, k, v), 2)

    padded_k, _ = _pad_to_two(k[:, 4:], dim=1)
    padded_v, _ = _pad_to_two(v[:, 4:], dim=1)
    for rank in range(2):
        def local(t):
            target, _ = _pad_to_two(t[:, 4:], dim=1)
            return torch.cat((t[:, :4], target[:, rank * 3:(rank + 1) * 3]), dim=1)

        gathered = iter((padded_k, padded_v))
        with _TwoRankPatch(rank, lambda tensor, dim: next(gathered)):
            actual = usp._sharded_attention(segments, 4, 5, {})(*_containers(local(q), local(k), local(v)), 2)
        valid = min(3, 5 - rank * 3)
        torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual[:, 4:4 + valid], expected[:, 4 + rank * 3:4 + rank * 3 + valid], rtol=1e-4, atol=1e-4)


def test_qwen21_sharded_forward_matches_block_causal_attention():
    torch.manual_seed(1)
    prefix = torch.randn(1, 4, 4)
    image = torch.randn(1, 4, 1, 5)
    segments = [(0, 2, torch.ones(2, 2, dtype=torch.bool).tril()), (2, 4, None), (4, 9, None)]
    model = _Model(prefix, segments, num_blocks=2)

    full = torch.cat((prefix, image.flatten(2).transpose(1, 2)), dim=1)
    block_targets = []
    for block in model.transformer_blocks:
        block_targets.append(_pad_to_two(full[:, 4:].unsqueeze(2), dim=1)[0])
        full = block(full, None, None, block_causal_attention(segments), 4, {})
    final_target = _pad_to_two(full[:, 4:], dim=1)[0]

    for rank in range(2):
        calls = []

        def all_gather(tensor, dim):
            calls.append(tensor)
            if len(calls) <= 2 * len(block_targets):
                # each block gathers the target K, then V; in this block both are its input
                return block_targets[(len(calls) - 1) // 2]
            valid = min(3, 5 - rank * 3)
            torch.testing.assert_close(tensor[:, :valid], final_target[:, rank * 3:rank * 3 + valid], rtol=1e-4, atol=1e-4)
            return final_target

        with _TwoRankPatch(rank, all_gather):
            result = usp.usp_dit_forward(model, image, torch.ones(1), prefix, transformer_options={})
        torch.testing.assert_close(result, full[:, 4:].transpose(1, 2).reshape(1, 4, 1, 5), rtol=1e-4, atol=1e-4)
        assert len(calls) == 5


def test_qwen21_block_replacement_receives_full_sequence():
    torch.manual_seed(2)
    prefix = torch.randn(1, 2, 4)
    image = torch.randn(1, 4, 1, 3)
    segments = [(0, 2, torch.ones(2, 2, dtype=torch.bool).tril()), (2, 5, None)]
    model = _Model(prefix, segments, num_blocks=1)
    calls = []

    def replace(args, extra):
        calls.append((args["img"].shape[1], args["prefix_len"], len(args["mod"])))
        out = extra["original_block"](args)
        out["img"] = out["img"] + 0.5
        return out

    full = torch.cat((prefix, image.flatten(2).transpose(1, 2)), dim=1)
    reference = replace({"img": full.clone(), "pe": None, "mod": (None,) * 5, "prefix_len": 2, "transformer_options": {}},
                        {"original_block": lambda args: {"img": model.transformer_blocks[0](args["img"], None, None,
                         block_causal_attention(segments), 2, {})}})["img"][:, 2:]
    calls.clear()

    for rank in range(2):
        gathered = []

        def all_gather(tensor, dim):
            gathered.append(tensor)
            if len(gathered) == 1:
                return _pad_to_two(full[:, 2:], dim=1)[0]
            valid = min(2, 3 - rank * 2)
            torch.testing.assert_close(tensor[:, :valid], reference[:, rank * 2:rank * 2 + valid], rtol=1e-4, atol=1e-4)
            return _pad_to_two(reference, dim=1)[0]

        with _TwoRankPatch(rank, all_gather):
            options = {"patches_replace": {"dit": {("single_block", 0): replace}}}
            output = usp.usp_dit_forward(model, image, torch.ones(1), prefix, transformer_options=options)
        torch.testing.assert_close(output, reference.transpose(1, 2).reshape(1, 4, 1, 3), rtol=1e-4, atol=1e-4)
        assert len(gathered) == 2
    assert calls == [(5, 2, 5), (5, 2, 5)]


def test_qwen21_prefix_cache_is_disabled_under_fsdp_and_usp():
    import comfy.supported_models
    from comfy import model_base
    from raylight.comfy_dist.model_patcher import _disable_prefix_cache
    from raylight.distributed_modules.usp import USPInjectRegistry

    config = comfy.supported_models.QwenImage21({"image_model": "qwen_image21", "in_channels": 64, "out_channels": 64,
                                                 "num_layers": 1, "attention_head_dim": 128, "num_attention_heads": 1,
                                                 "context_in_dim": 32, "mlp_ratio": 1})
    config.set_inference_dtype(torch.float32, None)
    for disable in (_disable_prefix_cache, lambda diffusion_model: USPInjectRegistry._REGISTRY[model_base.QwenImage21](
            None, SimpleNamespace(diffusion_model=diffusion_model))):
        model = config.get_model({})
        model.diffusion_model.prefix_cache_enabled = True
        assert model.diffusion_model.can_cache_prefix({})
        disable(model.diffusion_model)
        assert not model.diffusion_model.can_cache_prefix({})
        assert model.get_writable_model({}, (1, 64, 2, 2), {}, torch.device("cpu")) is None
