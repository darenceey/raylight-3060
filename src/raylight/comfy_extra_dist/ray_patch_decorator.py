import logging
import uuid

import ray
from functools import wraps


# Every worker holds one model, and Ray patch nodes replace it with a patched clone. A RAY_ACTORS value therefore names
# the model its node produced: the workers keep each patched clone under a token, and a node switches them back to the
# model of the RAY_ACTORS it was given before using it. A sampler then gets the patches of its own branch, the same way
# a ComfyUI MODEL output keeps its patches, even when ComfyUI reuses cached patch nodes, or runs another branch's
# patches in between.
MODEL_STATE_KEY = "model_state"
# How to build that model again: the loader call, then each Ray patch on top of it. A loader drops the models of the
# loader before it, so when a workflow loads two models into the same workers (Wan 2.2's high and low noise experts),
# a rerun that only reruns the samplers finds the other model there: ComfyUI keeps the loader outputs cached.
MODEL_RECIPE_KEY = "model_recipe"


def without_model_state(ray_actors):
    """ray_actors without the keys that name its model, to compare the actor sets of two RAY_ACTORS values."""
    return {**ray_actors, MODEL_STATE_KEY: None, MODEL_RECIPE_KEY: None}


def use_model_state(ray_actors):
    """Put every worker on the model the node that produced ray_actors left behind."""
    token = ray_actors.get(MODEL_STATE_KEY)
    if token is None:
        return
    found = ray.get([actor.activate_model_state.remote(token) for actor in ray_actors["workers"]])
    if all(found):
        return
    recipe = ray_actors.get(MODEL_RECIPE_KEY)
    if recipe is None:
        # The workers reloaded since (another loader, recovered workers): keep their current model
        logging.warning("[Raylight] Ray workers no longer hold the patched model this node was wired to; "
                        "using their current model. Re-run the Ray patch nodes in front of it to restore it.")
        return
    logging.info("[Raylight] Ray workers hold another model now: loading the model this node was wired to again")
    for step in recipe:
        step()
    ray.get([actor.save_model_state.remote(token) for actor in ray_actors["workers"]])


def save_model_state(ray_actors, loaded=False, replay=None):
    """Remember the workers' current model; returns the RAY_ACTORS value that names it.

    loaded marks the model a loader just produced, which the workers return to when a loader runs again.
    replay redoes what the node did to the workers' model (a loader: loads it, a Ray patch: patches it), so
    use_model_state can rebuild this model after another loader replaced it.
    """
    token = uuid.uuid4().hex
    ray.get([actor.save_model_state.remote(token, loaded=loaded) for actor in ray_actors["workers"]])
    recipe = () if loaded else ray_actors.get(MODEL_RECIPE_KEY)
    # A node that cannot say how to redo its change leaves a model that cannot be rebuilt
    recipe = None if recipe is None or replay is None else (*recipe, replay)
    return {**ray_actors, MODEL_STATE_KEY: token, MODEL_RECIPE_KEY: recipe}


def run_model_patch(ray_actors, fn, *args, **kwargs):
    """Apply fn(model, *args, **kwargs) to every worker's model; returns the call that applies it again."""
    gpu_workers = ray_actors["workers"]

    def apply():
        ray.get([actor.model_function_runner.remote(fn, *args, **kwargs) for actor in gpu_workers])

    apply()
    return apply


# Decorator to make a patch function Ray-distributable.
# Handles wrapping into _patch and Ray actor execution,
def ray_patch(patch_func):
    @wraps(patch_func)
    def wrapper(self, ray_actors, *args, **kwargs):
        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        use_model_state(ray_actors)
        replay = run_model_patch(ray_actors, _patch, *args, **kwargs)
        return (save_model_state(ray_actors, replay=replay),)
    return wrapper


# For nodes with return value, like produce float, int, or latent that still require model patcher.
def ray_patch_with_return(patch_func):
    @wraps(patch_func)
    def wrapper(self, ray_actors, *args, **kwargs):
        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        use_model_state(ray_actors)
        # Just need rank 0
        actor = ray_actors["workers"][0]
        value = ray.get(actor.model_function_runner_get_values.remote(_patch, *args, **kwargs))
        return value
    return wrapper
