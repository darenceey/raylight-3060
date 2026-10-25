import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
import comfy
import comfy.ldm.lightricks.model
from comfy.ldm.lightricks.model import apply_rotary_emb

GuideAttentionMask = getattr(comfy.ldm.lightricks.model, "GuideAttentionMask", None)  # ComfyUI before guide masks: None
from ..utils import pad_to_world_size

# Unpadded lengths of the sequences the attention keys come from, in transformer_options: {source: length}
SP_LENGTHS_KEY = "raylight_sp_lengths"

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


# A better solution is just to type.MethodType the x, context, and pe construction,
def pad_group_to_world_size(group, dim):
    """
    group: Tensor | list[Tensor] | tuple[Tensor]
    Returns: padded_group, orig_sizes
    """

    if torch.is_tensor(group):
        orig = group.size(dim)
        group, _ = pad_to_world_size(group, dim=dim)
        return group, orig

    elif isinstance(group, (list, tuple)):
        padded = []
        origs = []
        for g in group:
            o = g.size(dim)
            g, _ = pad_to_world_size(g, dim=dim)
            padded.append(g)
            origs.append(o)
        return type(group)(padded), origs

    else:
        return group, None


def pad_and_split_pe(pe, dim, sp_world_size, sp_rank):
    if torch.is_tensor(pe):
        pe, _ = pad_to_world_size(pe, dim=dim)
        return torch.chunk(pe, sp_world_size, dim=dim)[sp_rank]
    if isinstance(pe, tuple) and len(pe) == 2 and torch.is_tensor(pe[0]) and isinstance(pe[1], bool):
        return (pad_and_split_pe(pe[0], dim, sp_world_size, sp_rank), pe[1])
    if isinstance(pe, (list, tuple)):
        return type(pe)(pad_and_split_pe(item, dim, sp_world_size, sp_rank) for item in pe)
    return pe


def sp_chunk_group(group, sp_world_size, sp_rank, dim):
    if torch.is_tensor(group):
        return torch.chunk(group, sp_world_size, dim=dim)[sp_rank]

    elif isinstance(group, (list, tuple)):
        return type(group)(torch.chunk(g, sp_world_size, dim=dim)[sp_rank] for g in group)
    else:
        return group


def sp_gather_group(group, orig_sizes, dim):
    if torch.is_tensor(group):
        g = get_sp_group().all_gather(group.contiguous(), dim=dim)
        return g.narrow(dim, 0, orig_sizes)

    elif isinstance(group, (list, tuple)):
        out = []
        for g, o in zip(group, orig_sizes):
            g = get_sp_group().all_gather(g.contiguous(), dim=dim)
            g = g.narrow(dim, 0, o)
            out.append(g)
        return type(group)(out)

    else:
        return group


def process_usp_timestep(timestep, sp_rank, sp_world_size):
    if isinstance(timestep, (list, tuple)):
        return type(timestep)(
            process_usp_timestep(item, sp_rank, sp_world_size) for item in timestep
        )

    if torch.is_tensor(timestep) and timestep.ndim >= 3:
        orig_len = timestep.size(1)
        if orig_len == 0:
            return timestep

        padded_len = ((orig_len + sp_world_size - 1) // sp_world_size) * sp_world_size
        chunk_size = padded_len // sp_world_size
        start = sp_rank * chunk_size
        end = start + chunk_size

        if start >= orig_len:
            return timestep[:, -1:, :].expand(-1, chunk_size, -1).clone()
        if end <= orig_len:
            return timestep[:, start:end, :].clone()

        real_part = timestep[:, start:, :]
        padding = timestep[:, -1:, :].expand(-1, end - orig_len, -1)
        return torch.cat([real_part, padding], dim=1)

    if hasattr(timestep, "num_frames") and hasattr(timestep, "patches_per_frame"):
        total_len = timestep.num_frames * timestep.patches_per_frame
        if total_len == 0:
            return timestep.data[:, :0, :]

        padded_len = ((total_len + sp_world_size - 1) // sp_world_size) * sp_world_size
        chunk_size = padded_len // sp_world_size
        start = sp_rank * chunk_size
        end = start + chunk_size

        indices = torch.arange(start, end, device=timestep.data.device)
        valid_indices = torch.clamp(indices, max=total_len - 1)
        frame_indices = valid_indices // timestep.patches_per_frame
        return timestep.data[:, frame_indices, :]

    return timestep


def _sequence_lengths(x_orig, context_orig):
    """Unpadded token counts per key source: LTXV has video and text, LTXAV video, audio and their two text streams."""
    if isinstance(x_orig, list):
        return {"video": x_orig[0], "audio": x_orig[1], "video_context": context_orig[0], "audio_context": context_orig[1]}
    return {"video": x_orig, "video_context": context_orig}


def set_kv_sources(model):
    """Tag every attention with the sequence its keys come from (see SP_LENGTHS_KEY)."""
    sources = {"attn1": "video", "attn2": "video_context", "audio_attn1": "audio", "audio_attn2": "audio_context",
               "audio_to_video_attn": "audio", "video_to_audio_attn": "video"}
    for block in model.transformer_blocks:
        for name, source in sources.items():
            attention = getattr(block, name, None)
            if attention is not None:
                attention.raylight_kv_source = source


def _guide_attention_mask(model, x, transformer_options, merged_args):
    """ComfyUI's guide self-attention mask for the whole (unsplit) video sequence, or None (older ComfyUI, no guides)."""
    build = getattr(model, "_build_guide_self_attention_mask", None)
    return None if build is None else build(x, transformer_options, merged_args)


def _guide_attention(q, k, v, heads, guide_mask, row_start, attn_precision, transformer_options):
    """ComfyUI's _attention_with_guide_mask for this rank's queries, rows row_start onward of the video sequence.

    k and v hold every video token. As in ComfyUI the queries split into noisy, tracked-guide and remaining rows, so no
    dense (queries x keys) mask is built.
    """
    attention = comfy.ldm.modules.attention.optimized_attention
    guide_start = guide_mask.guide_start
    tracked_end = guide_start + guide_mask.tracked_count
    rows = q.shape[1]
    out = torch.empty_like(q)

    def run(lo, hi, mask):
        lo, hi = max(lo - row_start, 0), min(hi - row_start, rows)
        if lo < hi:
            out[:, lo:hi] = attention(q[:, lo:hi], k, v, heads, mask=mask, attn_precision=attn_precision,
                                      transformer_options=transformer_options)

    run(0, guide_start, guide_mask.noisy_mask)
    # the tracked mask's rows are the guide tokens' rows: take the ones this rank holds
    lo, hi = max(guide_start, row_start), min(tracked_end, row_start + rows)
    if lo < hi:
        run(lo, hi, guide_mask.tracked_mask[:, :, lo - guide_start:hi - guide_start])
    run(tracked_end, row_start + rows, None)
    return out


def usp_dit_forward(
    self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, denoise_mask=None, *args, **kwargs
):
    """
    Internal forward pass for LTX models.

    Args:
        x: Input tensor
        timestep: Timestep tensor
        context: Context tensor (e.g., text embeddings)
        attention_mask: Attention mask tensor
        frame_rate: Frame rate for temporal processing
        transformer_options: Additional options for transformer blocks
        keyframe_idxs: Keyframe indices for temporal processing
        **kwargs: Additional keyword arguments

    Returns:
        Processed output tensor
    """
    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if isinstance(x, list):
        input_dtype = x[0].dtype
        batch_size = x[0].shape[0]
    else:
        input_dtype = x.dtype
        batch_size = x.shape[0]
    # Process input
    merged_args = {**transformer_options, **kwargs}
    x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs, denoise_mask, **merged_args)
    merged_args.update(additional_args)

    # Prepare timestep and context
    timestep_prep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
    if not isinstance(timestep_prep, (list, tuple)):
        raise TypeError(f"Unexpected _prepare_timestep return type: {type(timestep_prep)!r}")

    if len(timestep_prep) == 3:
        timestep, embedded_timestep, prompt_timestep = timestep_prep
        if prompt_timestep is not None:
            merged_args["prompt_timestep"] = prompt_timestep
    elif len(timestep_prep) == 2:
        timestep, embedded_timestep = timestep_prep
    else:
        raise ValueError(f"Unexpected _prepare_timestep return length: {len(timestep_prep)}")

    context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)

    # Prepare attention mask and positional embeddings
    attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
    pe = self._prepare_positional_embeddings(pixel_coords, frame_rate, input_dtype)
    # Built for the whole sequence; the video self-attention applies it to its rank's query rows
    self_attention_mask = _guide_attention_mask(self, x, transformer_options, merged_args)

    # ===================== SP SPLIT ====================== #
    x, x_orig = pad_group_to_world_size(x, dim=1)
    context, context_orig = pad_group_to_world_size(context, dim=1)
    pe = pad_and_split_pe(pe, dim=1, sp_world_size=sp_world_size, sp_rank=sp_rank)
    transformer_options = {**transformer_options, SP_LENGTHS_KEY: _sequence_lengths(x_orig, context_orig)}

    x = sp_chunk_group(x, sp_world_size, sp_rank, dim=1)
    context = sp_chunk_group(context, sp_world_size, sp_rank, dim=1)
    timestep = process_usp_timestep(timestep, sp_rank, sp_world_size)
    if "prompt_timestep" in merged_args:
        merged_args["prompt_timestep"] = process_usp_timestep(merged_args["prompt_timestep"], sp_rank, sp_world_size)

    # Process transformer blocks
    if self_attention_mask is not None:
        merged_args["self_attention_mask"] = self_attention_mask
    x = self._process_transformer_blocks(x, context, attention_mask, timestep, pe, transformer_options=transformer_options, **merged_args)

    # ===================== SP GATHER ===================== #
    x = sp_gather_group(x, x_orig, dim=1)

    # Process output
    x = self._process_output(x, embedded_timestep, keyframe_idxs, **merged_args)
    return x


def usp_cross_attn_forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}, *args, **kwargs):
    self_attn = context is None
    context = x if context is None else context
    v = self.to_v(context)

    if self_attn and transformer_options.get("stg_skip_self_attn", False):
        out = v
    else:
        q = self.q_norm(self.to_q(x))
        k = self.k_norm(self.to_k(context))

        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

        kv_length = transformer_options.get(SP_LENGTHS_KEY, {}).get(getattr(self, "raylight_kv_source", None))
        guide_mask = mask if GuideAttentionMask is not None and isinstance(mask, GuideAttentionMask) else None
        if kv_length is not None and (guide_mask is not None or kv_length % get_sequence_parallel_world_size() != 0):
            # The key sequence was zero-padded to split it, and Ulysses attention would attend to the padding, or
            # guide tokens need their attention weights: keep the queries local and attend to the real keys of every
            # rank instead.
            k = get_sp_group().all_gather(k.contiguous(), dim=1).narrow(1, 0, kv_length)
            v = get_sp_group().all_gather(v.contiguous(), dim=1).narrow(1, 0, kv_length)
            if guide_mask is not None:
                out = _guide_attention(q, k, v, self.heads, guide_mask, get_sequence_parallel_rank() * q.shape[1],
                                       self.attn_precision, transformer_options)
            else:
                out = comfy.ldm.modules.attention.optimized_attention(
                    q, k, v, self.heads, mask=mask, attn_precision=self.attn_precision, transformer_options=transformer_options
                )
        else:
            out = xfuser_optimized_attention(
                q,
                k,
                v,
                self.heads,
                attn_precision=self.attn_precision,
            )

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)
