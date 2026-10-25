import torch

from raylight.distributed_modules.cfg_utils import chunk_cfg_batch, direct_cfg_split, gather_cfg_batch, patch_direct_cfg_forward


def cfg_parallel_forward(self, x, t, context, transformer_options={}, **kwargs):
    split = direct_cfg_split(transformer_options)
    if split is None:
        return self._raylight_cfg_original_forward(x, t, context, transformer_options=transformer_options, **kwargs)
    cfg_rank, cfg_world_size, transformer_options = split

    def chunk(value):
        return chunk_cfg_batch(value, cfg_world_size, cfg_rank)

    x, t, context = chunk(x), chunk(t), chunk(context)
    kwargs = {key: chunk(value) if isinstance(value, torch.Tensor) and value.shape[:1] == (len(x) * cfg_world_size,) else value
              for key, value in kwargs.items()}

    # HunYuanDiTPlain swaps the halves of any batch of two or more; that is a no-op for a cond/uncond pair of the same
    # latent, but pairs each latent of a rank's own batch with the wrong condition. Run those items one at a time.
    outputs = []
    for i in range(len(x)):
        item_kwargs = {key: value[i:i + 1] if isinstance(value, torch.Tensor) and value.shape[:1] == (len(x),) else value
                       for key, value in kwargs.items()}
        outputs.append(self._raylight_cfg_original_forward(x[i:i + 1], t[i:i + 1], context[i:i + 1],
                                                           transformer_options=transformer_options, **item_kwargs))
    return gather_cfg_batch(torch.cat(outputs, dim=0))


def patch_cfg_forward(model):
    patch_direct_cfg_forward(model, cfg_parallel_forward)
