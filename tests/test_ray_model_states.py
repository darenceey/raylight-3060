"""Ray patch nodes and samplers keep each RAY_ACTORS value tied to the patched model its node produced."""
import collections
import types

import pytest

import raylight.comfy_extra_dist.ray_patch_decorator as decorator
from raylight.distributed_worker.ray_worker import RayWorker


class _Model:
    def __init__(self, patches=()):
        self.patches = tuple(patches)

    def clone(self):
        return _Model(self.patches)


class _Actor:
    """A Ray actor handle whose .remote() calls run the real RayWorker method in this process."""

    def __init__(self, worker):
        self.worker = worker

    def __getattr__(self, name):
        method = getattr(self.worker, name)
        return types.SimpleNamespace(remote=method)


def _worker():
    worker = RayWorker.__new__(RayWorker)  # skips process-group setup; only the model state methods run
    worker.model = _Model()
    worker.model_states = collections.OrderedDict()
    worker.loaded_model = None
    return worker


class _PatchNode:
    @decorator.ray_patch
    def patch(self, model, name):
        m = model.clone()
        m.patches = m.patches + (name,)
        return m


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.setattr(decorator, "ray", types.SimpleNamespace(get=lambda value: value))
    workers = [_worker(), _worker()]
    return workers, {"workers": [_Actor(w) for w in workers]}


def _sample(workers, ray_actors):
    # what a Ray sampler does before sampling
    decorator.use_model_state(ray_actors)
    return [w.model.patches for w in workers]


def test_cached_stage_patches_come_back_on_a_rerun(workers):
    workers, init = workers
    loaded = decorator.save_model_state(init, loaded=True)
    node = _PatchNode()

    # first run, as in the TRELLIS.2 template: structure patches, sampler, shape patches on its output, sampler
    (structure,) = node.patch(loaded, "shift 5")
    assert _sample(workers, structure) == [("shift 5",)] * 2
    (shape,) = node.patch(structure, "shift 3")
    assert _sample(workers, shape) == [("shift 5", "shift 3")] * 2

    # second run with a new image: ComfyUI reuses the cached structure patch node and only re-runs the sampler
    assert _sample(workers, structure) == [("shift 5",)] * 2
    assert _sample(workers, shape) == [("shift 5", "shift 3")] * 2


def test_branches_off_one_model_do_not_see_each_other(workers):
    workers, init = workers
    loaded = decorator.save_model_state(init, loaded=True)
    node = _PatchNode()
    (a,) = node.patch(loaded, "a")
    (b,) = node.patch(loaded, "b")
    assert _sample(workers, a) == [("a",)] * 2
    assert _sample(workers, b) == [("b",)] * 2
    assert _sample(workers, loaded) == [()] * 2


def test_a_loader_rerun_drops_earlier_patches(workers):
    workers, init = workers
    loaded = decorator.save_model_state(init, loaded=True)
    (patched,) = _PatchNode().patch(loaded, "a")
    _sample(workers, patched)
    # RayUNETLoader finding the model already loaded
    for w in workers:
        w.reset_to_loaded_model()
    reloaded = decorator.save_model_state(init, loaded=True)
    assert _sample(workers, reloaded) == [()] * 2
    # the old patched clones are gone, so they cannot keep a replaced model alive
    assert all(list(w.model_states) == [reloaded[decorator.MODEL_STATE_KEY]] for w in workers)


def test_unknown_state_keeps_the_current_model(workers, caplog):
    workers, init = workers
    (patched,) = _PatchNode().patch(decorator.save_model_state(init, loaded=True), "a")
    for w in workers:
        w._forget_model_states()  # the workers loaded another model since
    _sample(workers, {**init, decorator.MODEL_STATE_KEY: "gone"})
    assert [w.model.patches for w in workers] == [("a",)] * 2
    assert "no longer hold the patched model" in caplog.text


def test_values_without_a_state_keep_the_old_behaviour(workers):
    workers, init = workers
    (patched,) = _PatchNode().patch(init, "a")
    assert _sample(workers, init) == [("a",)] * 2
    assert decorator.MODEL_STATE_KEY in patched
