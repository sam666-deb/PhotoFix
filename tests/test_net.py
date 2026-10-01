import numpy as np
import pytest
import torch
from skimage import data

from photofix import enhance
from photofix.analyzer import DEFECTS
from photofix.net import VIEW_SIZE, DefectNet, DLAnalyzer, make_views


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """An untrained model is enough to test loading and the plumbing."""
    path = tmp_path_factory.mktemp("ckpt") / "analyzer.pt"
    torch.manual_seed(0)
    torch.save({"model": DefectNet(pretrained=False).state_dict(), "defects": list(DEFECTS)}, path)
    return path


@pytest.mark.parametrize("shape", [(512, 512, 3), (3000, 4000, 3), (150, 900, 3)])
def test_make_views_shapes(shape):
    rgb = np.random.default_rng(0).random(shape, dtype=np.float32)
    for rng in (None, np.random.default_rng(1)):
        g, l = make_views(rgb, rng)
        assert g.shape == l.shape == (VIEW_SIZE, VIEW_SIZE, 3)
        assert g.dtype == l.dtype == np.float32


def test_center_crop_is_deterministic():
    rgb = np.random.default_rng(0).random((600, 800, 3), dtype=np.float32)
    assert np.array_equal(make_views(rgb)[1], make_views(rgb)[1])


def test_dl_analyzer_plugs_into_enhance(checkpoint):
    net = DLAnalyzer(checkpoint, device=torch.device("cpu"))
    img = data.astronaut().astype(np.float32) / 255.0
    out, analysis, _ = enhance(img, net=net)
    assert analysis.source == "dl"
    assert set(analysis.scores) == set(DEFECTS)
    assert all(0.0 <= v <= 1.0 for v in analysis.scores.values())
    assert out.shape == img.shape


def test_rejects_mismatched_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model": {}, "defects": ["noise"]}, path)
    with pytest.raises(ValueError):
        DLAnalyzer(path, device=torch.device("cpu"))
