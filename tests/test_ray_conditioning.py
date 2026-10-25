import torch

from raylight.comfy_extra_dist.nodes_custom_sampler import _make_ray_guider, strip_host_only_conds


def _pixal3d_conditioning():
    pack = {"feature_map": torch.zeros(1, 4, 8, 8)}
    return [[torch.zeros(1, 3, 4), {"embeds": torch.ones(1, 3, 4), "trellis2_proj_feats": torch.ones(5, 2),
                                    "proj_feat_pack": pack}]]


def test_feature_pack_stays_on_the_host():
    conditioning = _pixal3d_conditioning()
    stripped = strip_host_only_conds(conditioning)
    assert set(stripped[0][1]) == {"embeds", "trellis2_proj_feats"}
    assert stripped[0][0] is conditioning[0][0]
    # the caller's conditioning keeps its pack: later stage nodes on the host still read it
    assert "proj_feat_pack" in conditioning[0][1]


def test_conditioning_without_host_only_entries_passes_through():
    conditioning = [[torch.zeros(1, 3, 4), {"embeds": torch.ones(1, 3, 4)}]]
    assert strip_host_only_conds(conditioning)[0][1] is conditioning[0][1]
    assert strip_host_only_conds(None) is None


def test_guiders_carry_stripped_conditioning():
    guider = _make_ray_guider({"workers": []}, "cfg", positive=_pixal3d_conditioning(), negative=_pixal3d_conditioning(), cfg=7.5)
    assert "proj_feat_pack" not in guider["positive"][0][1]
    assert "proj_feat_pack" not in guider["negative"][0][1]
    assert guider["cfg"] == 7.5
