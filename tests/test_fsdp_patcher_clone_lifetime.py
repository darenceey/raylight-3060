"""A dead FSDPModelPatcher clone must not unregister the clone that is sampling.

Clones share the model. The collector can finalize an old clone in the middle of another clone's run, and its
__del__ used to clear model.current_patcher regardless, so ComfyUI's cond batching found None there and crashed:

    AttributeError: 'NoneType' object has no attribute 'prepare_state'
"""
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

from raylight.comfy_dist.model_patcher import FSDPModelPatcher


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Linear(4, 4)
        self.current_patcher = None


def _patcher():
    return FSDPModelPatcher(_Root(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def test_a_dead_clone_leaves_the_sampling_clone_registered():
    old = _patcher()
    sampling = old.clone()
    model = old.model
    model.current_patcher = sampling
    old.__del__()
    assert model.current_patcher is sampling


def test_a_dead_patcher_drops_its_own_registration():
    patcher = _patcher()
    model = patcher.model
    model.current_patcher = patcher
    patcher.__del__()
    assert model.current_patcher is None
