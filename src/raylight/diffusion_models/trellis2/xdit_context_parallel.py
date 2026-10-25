import types

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy.ldm.trellis2.model import (
    MultiHeadAttention,
    SLatFlowModel,
    SparseMultiHeadAttention,
    SparseStructureFlowModel,
    dense_attention,
    sparse_attention,
)
from comfy.ldm.trellis2.vae import SparseTensor, VarLenTensor, sparse_cat
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

from ..utils import gather_sequence, shard_sequence

# TRELLIS.2 runs four flow models: the dense 16^3 structure model and three sparse voxel models (shape 512, shape 1024,
# texture). Each rank keeps a shard of every item's tokens. Self-attention is the only op that mixes tokens, so it
# gathers the K/V of the whole sequence while its queries stay local; cross-attention reads the image condition, which
# every rank holds in full. The forwards pass the shard layout to self-attention through transformer_options.

SP_PLAN_KEY = "raylight_sp_plan"
SP_LEN_KEY = "raylight_sp_len"


class _SparseShardPlan:
    """Which voxels this rank holds, and where every voxel lands in the gathered shards.

    Every item is split on its own into world_size equal shards, so the rank shards line up item by item and the
    native per-item layout (CFG copies, several seeds with different voxel counts) is kept. A shard that runs past the
    end of its item repeats the item's last voxel; those copies are never gathered back.
    """

    def __init__(self, layout, world_size, rank, device):
        lengths = [item.stop - item.start for item in layout]
        shard_lengths = [-(-length // world_size) for length in lengths]
        self.local_len = sum(shard_lengths)
        self.full_layout = layout
        self.local_layout = []
        local_index, gather_index = [], []
        offset = 0
        for item, length, shard_length in zip(layout, lengths, shard_lengths):
            positions = torch.arange(rank * shard_length, (rank + 1) * shard_length, device=device)
            local_index.append(item.start + positions.clamp(max=length - 1))
            tokens = torch.arange(length, device=device)
            gather_index.append(torch.div(tokens, shard_length, rounding_mode="floor") * self.local_len + offset + tokens % shard_length)
            self.local_layout.append(slice(offset, offset + shard_length))
            offset += shard_length
        self.local_index = torch.cat(local_index)
        self.gather_index = torch.cat(gather_index)

    def gather(self, feats):
        return get_sp_group().all_gather(feats.contiguous(), dim=0)[self.gather_index]


def _shard_sparse_context(context, plan):
    # Pixal3D projected features hold one row per voxel, in the order of x.feats
    if not isinstance(context, dict):
        return context
    context = dict(context)
    for key in ("proj", "proj_semantic", "proj_color"):
        if isinstance(context.get(key), torch.Tensor):
            context[key] = context[key][plan.local_index.to(context[key].device)]
    return context


def _shard_dense_context(context):
    # The structure model's projected features are [B, 16^3, C], one row per grid cell
    if not isinstance(context, dict):
        return context
    context = dict(context)
    for key in ("proj", "proj_semantic", "proj_color"):
        if isinstance(context.get(key), torch.Tensor):
            context[key] = shard_sequence(context[key], dim=1)[0]
    return context


def usp_sparse_self_attn_forward(self, x, context=None, transformer_options=None):
    qkv = self._linear(self.to_qkv, x)
    qkv = self._fused_pre(qkv, num_fused=3)
    q, k, v = qkv.unbind(dim=-3)
    if self.qk_rms_norm:
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
    del qkv
    q, k = self.rope(q, k)
    plan = transformer_options[SP_PLAN_KEY]
    k = VarLenTensor(plan.gather(k.feats), plan.full_layout)
    v = VarLenTensor(plan.gather(v.feats), plan.full_layout)
    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    h = sparse_attention(q, k, v, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
    h = self._reshape_chs(h, (-1,))
    h = self._linear(self.to_out, h)
    return h


def usp_slat_forward(
    self,
    x,
    t,
    cond,
    concat_cond=None,
    transformer_options=None,
    **kwargs,
):
    if concat_cond is not None:
        x = sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = VarLenTensor.from_tensor_list(cond)

    # ===================== SP SPLIT ====================== #
    plan = _SparseShardPlan(x.layout, get_sequence_parallel_world_size(), get_sequence_parallel_rank(), x.device)
    local = SparseTensor(feats=x.feats[plan.local_index], coords=x.coords[plan.local_index], shape=x.shape)
    local.register_spatial_cache("layout", plan.local_layout)
    cond = _shard_sparse_context(cond, plan)
    transformer_options = {**(transformer_options or {}), SP_PLAN_KEY: plan}

    h = self.input_layer(local)
    t_emb = self.t_embedder(t, out_dtype=t.dtype)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)

    for block in self.blocks:
        h = block(h, t_emb, cond, transformer_options=transformer_options)

    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h)

    # ===================== SP GATHER ===================== #
    return x.replace(plan.gather(h.feats))


def usp_dense_self_attn_forward(self, x, context=None, phases=None, transformer_options=None):
    B, L, C = x.shape
    qkv = self.to_qkv(x)
    qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
    q, k, v = qkv.unbind(dim=2)
    if self.qk_rms_norm:
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
    assert phases is not None, "Phases must be provided for RoPE"
    f_cis = comfy.model_management.cast_to(phases, device=q.device).unsqueeze(0).unsqueeze(2)
    del qkv
    q, k = apply_rope(q, k, f_cis)
    seq_len = transformer_options[SP_LEN_KEY]
    k = gather_sequence(k, seq_len, dim=1)
    v = gather_sequence(v, seq_len, dim=1)
    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    h = dense_attention(q, k, v, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
    h = h.reshape(B, L, -1)
    h = self.to_out(h)
    return h


def usp_ss_forward(self, x, t, cond, transformer_options=None):
    x = x.view(x.shape[0], self.in_channels, *[self.resolution] * 3)

    h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()

    # ===================== SP SPLIT ====================== #
    h, seq_len = shard_sequence(h, dim=1)
    rope_phases = shard_sequence(self.rope_phases, dim=0)[0]
    cond = _shard_dense_context(cond)
    transformer_options = {**(transformer_options or {}), SP_LEN_KEY: seq_len}

    h = self.input_layer(h)
    t_emb = self.t_embedder(t, out_dtype=t.dtype)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    for block in self.blocks:
        h = block(h, t_emb, cond, rope_phases, transformer_options=transformer_options)
    h = F.layer_norm(h, h.shape[-1:])
    h = self.out_layer(h)

    # ===================== SP GATHER ===================== #
    h = gather_sequence(h, seq_len, dim=1)

    h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()

    return h


def inject_usp(model):
    for flow_model in (getattr(model, name, None) for name in ("structure_model", "img2shape_512", "img2shape", "shape2txt")):
        if isinstance(flow_model, SLatFlowModel):
            flow_model.forward = types.MethodType(usp_slat_forward, flow_model)
            for block in flow_model.blocks:
                assert isinstance(block.self_attn, SparseMultiHeadAttention)
                block.self_attn.forward = types.MethodType(usp_sparse_self_attn_forward, block.self_attn)
        elif isinstance(flow_model, SparseStructureFlowModel):
            flow_model.forward = types.MethodType(usp_ss_forward, flow_model)
            for block in flow_model.blocks:
                assert isinstance(block.self_attn, MultiHeadAttention)
                block.self_attn.forward = types.MethodType(usp_dense_self_attn_forward, block.self_attn)
