import torch

import comfy.model_prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.ldm.qwen_image21.model import _split_rows, block_causal_attention
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

from ..utils import pad_to_world_size


def _sharded_attention(segments, prefix_len, target_len, transformer_options):
    # Local rows are [prefix, target shard]. The prefix is replicated on every rank and keeps its block-causal attention,
    # the target shard attends to the prefix and the whole target gathered from the SP group.
    prefix_attention = block_causal_attention(segments[:-1], transformer_options)

    def attn(q, k, v, heads, preferred_attention=None):
        q, k, v = q.take(), k.take(), v.take()
        group = get_sp_group()
        target_k = group.all_gather(k[:, prefix_len:].contiguous(), dim=1)[:, :target_len]
        target_v = group.all_gather(v[:, prefix_len:].contiguous(), dim=1)[:, :target_len]
        out = optimized_attention(AttentionTensorContainer(q[:, prefix_len:].flatten(2)),
                                  AttentionTensorContainer(torch.cat((k[:, :prefix_len], target_k), dim=1).flatten(2)),
                                  AttentionTensorContainer(torch.cat((v[:, :prefix_len], target_v), dim=1).flatten(2)),
                                  heads, transformer_options=transformer_options, preferred_attention=preferred_attention)
        if not prefix_len:
            return out
        prefix = prefix_attention(AttentionTensorContainer(q[:, :prefix_len]), AttentionTensorContainer(k[:, :prefix_len]),
                                  AttentionTensorContainer(v[:, :prefix_len]), heads, preferred_attention=preferred_attention)
        return torch.cat((prefix, out), dim=1)
    return attn


def usp_dit_forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
    b, _, h, w = x.shape
    dtype = x.dtype
    ref_latents = list(ref_latents or [])
    image_slots = list(image_slots or [])

    hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
    prefix_len = hidden_states.shape[1] - h * w
    patches = transformer_options.get("patches", {})
    for p in patches.get("post_input", []):
        out = p({"img": hidden_states, "pe": pe, "transformer_options": transformer_options})
        hidden_states, pe = out["img"], out.get("pe", pe)

    t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
    temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
    scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
    mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))

    target_len = hidden_states.shape[1] - prefix_len
    rank = get_sequence_parallel_rank()
    world_size = get_sequence_parallel_world_size()

    def shard(full):
        # Each block runs once per rank on [prefix, target shard], so an FSDP rank gathers the block's weights once.
        target, _ = pad_to_world_size(full[:, prefix_len:], dim=1)
        return torch.cat((full[:, :prefix_len], torch.chunk(target, world_size, dim=1)[rank]), dim=1)

    def gather_target(local):
        return get_sp_group().all_gather(local[:, prefix_len:].contiguous(), dim=1)[:, :target_len]

    local_states = shard(hidden_states)
    local_pe = shard(pe)

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(self.transformer_blocks)
    transformer_options["block_type"] = "single"
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), x.device, transformer_options)
    for i, block in enumerate(self.transformer_blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, dtype)
        transformer_options["block_index"] = i
        if (("single_block", i) in blocks_replace or patches.get("single_block") or patches.get("attn1_patch") or
                transformer_options.get("optimized_attention_override") is not None or block.attn.comfy_attention.function is not None):
            hidden_states = torch.cat((local_states[:, :prefix_len], gather_target(local_states)), dim=1)
            attn_fn = block_causal_attention(segments, transformer_options, block_index=i, prefix_len=prefix_len)
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], mod, args["pe"], attn_fn, prefix_len, args["transformer_options"])}
                args = {"img": hidden_states, "vec": temb, "pe": pe, "mod": mod, "attn_fn": attn_fn, "prefix_len": prefix_len,
                        "transformer_options": transformer_options}
                hidden_states = blocks_replace[("single_block", i)](args, {"original_block": block_wrap})["img"]
            else:
                hidden_states = block(hidden_states, mod, pe, attn_fn, prefix_len, transformer_options)
            for p in patches.get("single_block", []):
                hidden_states = p({"img": hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})["img"]
            local_states = shard(hidden_states)
            continue

        attn_fn = _sharded_attention(segments, prefix_len, target_len, transformer_options)
        local_states = block(local_states, mod, local_pe, attn_fn, prefix_len, transformer_options)

    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
    target = gather_target(local_states)
    target = self.norm_out(target, temb[:-1])
    target = self.proj_out(target)
    return target.transpose(1, 2).reshape(b, self.out_channels, h, w)
