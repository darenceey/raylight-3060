"""Every Ray loader patches its freshly loaded model for the degrees the initializer set, through one helper."""
import ast
import types
from pathlib import Path

import pytest

import raylight.distributed_worker.ray_worker as ray_worker

ROOT = Path(__file__).parents[1] / "src/raylight"


class _Actor:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("patch_"):
            raise AttributeError(name)
        return types.SimpleNamespace(remote=lambda: self.calls.append(name))


def _parallel_dict(ulysses=1, ring=1, cfg=1, pipefusion=False):
    return {"is_xdit": ulysses * ring * cfg > 1 or pipefusion, "ulysses_degree": ulysses, "ring_degree": ring,
            "cfg_degree": cfg, "pipefusion_enabled": pipefusion}


@pytest.mark.parametrize("degrees, expected", [
    ({}, []),
    ({"ulysses": 2}, ["patch_usp"]),
    ({"ring": 2}, ["patch_usp"]),
    ({"cfg": 2}, ["patch_cfg"]),
    ({"ulysses": 2, "cfg": 2}, ["patch_usp", "patch_cfg"]),
    ({"pipefusion": True}, ["patch_pipefusion"]),
])
def test_patches_follow_the_degrees(monkeypatch, degrees, expected):
    monkeypatch.setattr(ray_worker, "ray", types.SimpleNamespace(get=lambda value: value))
    actors = [_Actor(), _Actor()]
    ray_worker.patch_parallelism(actors, _parallel_dict(**degrees))
    assert [actor.calls for actor in actors] == [expected, expected]


def _calls(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    return {node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


@pytest.mark.parametrize("path, class_name, function_name", [
    (ROOT / "nodes.py", "RayUNETLoader", "load_ray_unet"),
    # GGUF models skipped CFG parallel before this helper: every CFG rank sampled the whole batch
    (ROOT / "expansion/comfyui_gguf/nodes.py", "RayGGUFLoader", "load_ray_unet"),
])
def test_loaders_use_the_shared_helper(path, class_name, function_name):
    assert "patch_parallelism" in _calls(path, class_name, function_name)
