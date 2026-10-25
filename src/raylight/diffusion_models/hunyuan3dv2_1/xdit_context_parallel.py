import torch

from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

from ..utils import gather_sequence, shard_sequence

# Each rank runs the HunYuanDiT blocks on a shard of [time token, latents]. Everything but self-attention is per token
# (cross-attention reads the full image condition on every rank), so only attn1 needs the K/V of the whole sequence.
# HunYuanDiTBlock does not pass transformer_options down, so usp_dit_forward records the sequence length in a dict
# that the injection shares between the model and every attn1 as `raylight_sp_state`.


def usp_attention_forward(self, x):
    B, N, _ = x.shape

    query = self.to_q(x)
    key = self.to_k(x)
    value = self.to_v(x)

    qkv_combined = torch.cat((query, key, value), dim=-1)
    split_size = qkv_combined.shape[-1] // self.num_heads // 3

    qkv = qkv_combined.view(B, -1, self.num_heads, split_size * 3)
    query, key, value = torch.split(qkv, split_size, dim=-1)

    query = query.reshape(B, N, self.num_heads, self.head_dim)
    key = key.reshape(B, N, self.num_heads, self.head_dim)
    value = value.reshape(B, N, self.num_heads * self.head_dim)

    query = self.q_norm(query)
    key = self.k_norm(key)

    seq_len = self.raylight_sp_state["seq_len"]
    key = gather_sequence(key.reshape(B, N, -1), seq_len, dim=1)
    value = gather_sequence(value, seq_len, dim=1)

    query, key, value = AttentionTensorContainer(query.reshape(B, N, -1)), AttentionTensorContainer(key), AttentionTensorContainer(value)
    del qkv, qkv_combined
    x = optimized_attention(
        query,
        key,
        value,
        heads=self.num_heads,
        low_precision_attention=False, preferred_attention=self.comfy_attention,
    )

    x = self.out_proj(x)
    return x


def usp_dit_forward(self, x, t, context, transformer_options = {}, **kwargs):

    x = x.movedim(-1, -2)

    swap_cfg_halves = context.shape[0] >= 2

    if swap_cfg_halves:
        first_half, second_half = context.chunk(2, dim = 0)
        context = torch.cat([second_half, first_half], dim = 0)

    main_condition = context

    t = 1.0 - t

    time_embedded = self.t_embedder(t, condition = kwargs.get('guidance_cond'))

    x = x.to(dtype = next(self.x_embedder.parameters()).dtype)
    x_embedded = self.x_embedder(x)

    combined = torch.cat([time_embedded, x_embedded], dim=1)

    # ===================== SP SPLIT ====================== #
    sp_state = self.raylight_sp_state
    combined, sp_state["seq_len"] = shard_sequence(combined, dim=1)

    def block_wrap(args):
        return block(
            args["x"],
            args["t"],
            args["cond"],
            skip_tensor=args.get("skip"),)

    skip_stack = []
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    for idx, block in enumerate(self.blocks):
        if idx <= self.depth // 2:
            skip_input = None
        else:
            skip_input = skip_stack.pop()

        if ("block", idx) in blocks_replace:

            combined = blocks_replace[("block", idx)](
                {
                    "x": combined,
                    "t": time_embedded,
                    "cond": main_condition,
                    "skip": skip_input,
                },
                {"original_block": block_wrap},
            )
        else:
            combined = block(combined, time_embedded, main_condition, skip_tensor=skip_input)

        if idx < self.depth // 2:
            skip_stack.append(combined)

    # ===================== SP GATHER ===================== #
    combined = gather_sequence(combined, sp_state["seq_len"], dim=1)

    output = self.final_layer(combined)
    output =  output.movedim(-2, -1) * (-1.0)

    if swap_cfg_halves:
        first_half, second_half = output.chunk(2, dim = 0)
        output = torch.cat([second_half, first_half], dim = 0)

    return output
