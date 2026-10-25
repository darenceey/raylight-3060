import torch

from comfy.ldm.flux.layers import timestep_embedding

from ..utils import gather_sequence, shard_sequence

# Hunyuan3D 2.0 runs ComfyUI's own Flux double/single blocks. Each rank holds a shard of the text and latent tokens, and
# the attn1 patch below hands every block the K/V of the whole sequence while its queries stay local.

SP_LENGTHS_KEY = "raylight_sp_lengths"


def _gather_kv(q, k, v, pe=None, attn_mask=None, extra_options={}):
    lengths = extra_options[SP_LENGTHS_KEY]
    if len(lengths) == 1:
        return {"k": gather_sequence(k, lengths[0], dim=2), "v": gather_sequence(v, lengths[0], dim=2)}

    # Double blocks attend over [txt shard, img shard]; the two streams were padded and split separately.
    txt_local = extra_options["img_slice"][0]
    txt_len, img_len = lengths

    def gather(t):
        return torch.cat((gather_sequence(t[:, :, :txt_local], txt_len, dim=2), gather_sequence(t[:, :, txt_local:], img_len, dim=2)), dim=2)

    return {"k": gather(k), "v": gather(v)}


def usp_dit_forward(self, x, timestep, context, guidance=None, transformer_options={}, **kwargs):
    x = x.movedim(-1, -2)
    timestep = 1.0 - timestep
    txt = context
    img = self.latent_in(x)

    vec = self.time_in(timestep_embedding(timestep, 256, self.max_period).to(dtype=img.dtype))
    if self.guidance_in is not None:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256, self.max_period).to(img.dtype))

    txt = self.cond_in(txt)
    pe = None
    attn_mask = None

    transformer_options = transformer_options.copy()
    patches = dict(transformer_options.get("patches", {}))
    # Last, so every other attn1 patch still sees this rank's own tokens
    patches["attn1_patch"] = [*patches.get("attn1_patch", []), _gather_kv]
    transformer_options["patches"] = patches

    # ===================== SP SPLIT ====================== #
    img, img_len = shard_sequence(img, dim=1)
    txt, txt_len = shard_sequence(txt, dim=1)
    transformer_options[SP_LENGTHS_KEY] = (txt_len, img_len)

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               transformer_options=args["transformer_options"])
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask,
                             transformer_options=transformer_options)

    # ===================== SP GATHER ===================== #
    img = gather_sequence(img, img_len, dim=1)
    txt = gather_sequence(txt, txt_len, dim=1)

    img = torch.cat((txt, img), 1)

    # ===================== SP SPLIT ====================== #
    img, joint_len = shard_sequence(img, dim=1)
    transformer_options[SP_LENGTHS_KEY] = (joint_len,)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args["transformer_options"])
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

    # ===================== SP GATHER ===================== #
    img = gather_sequence(img, joint_len, dim=1)

    img = img[:, txt_len:, ...]
    img = self.final_layer(img, vec)
    return img.movedim(-2, -1) * (-1.0)
