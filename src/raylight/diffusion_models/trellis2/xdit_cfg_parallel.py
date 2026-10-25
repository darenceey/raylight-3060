from raylight.distributed_modules.cfg_utils import chunk_cfg_batch, direct_cfg_split, gather_cfg_batch, patch_direct_cfg_forward


def cfg_parallel_forward(self, x, timestep, context, **kwargs):
    split = direct_cfg_split(kwargs.get("transformer_options", {}))
    if split is None:
        return self._raylight_cfg_original_forward(x, timestep, context, **kwargs)
    cfg_rank, cfg_world_size, transformer_options = split

    def chunk(value):
        return chunk_cfg_batch(value, cfg_world_size, cfg_rank)

    # Only these follow the cond/uncond batch. The trellis2_* stage inputs (coords, counts, shape latent, projected
    # features) are shared by both, and Trellis2.forward repeats them to whatever batch it gets.
    kwargs["transformer_options"] = transformer_options
    if "embeds" in kwargs:
        kwargs["embeds"] = chunk(kwargs["embeds"])
    out = self._raylight_cfg_original_forward(chunk(x), chunk(timestep), chunk(context), **kwargs)
    return gather_cfg_batch(out)


def patch_cfg_forward(model):
    patch_direct_cfg_forward(model, cfg_parallel_forward)
