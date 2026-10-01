import numpy as np
import pytest
import torch
from skimage import data

from photofix import enhance
from photofix.lut import LUT_DIM, N_BASIS, LUTEnhancer, LUTNet, apply_lut, identity_lut, lut_regularization


@pytest.fixture(scope="module")
def astronaut():
    return data.astronaut().astype(np.float32) / 255.0


@pytest.fixture(scope="module")
def untrained_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("lut") / "lut.pt"
    torch.save({"model": LUTNet().state_dict(), "n_basis": N_BASIS, "dim": LUT_DIM}, path)
    return path


def test_identity_lut_is_a_no_op():
    img = torch.rand(2, 3, 40, 60)
    lut = identity_lut()[None].expand(2, -1, -1, -1, -1)
    assert torch.allclose(apply_lut(img, lut), img, atol=1e-5)


def test_untrained_model_is_a_no_op():
    img = torch.rand(1, 3, 32, 32)
    lut = LUTNet()(torch.rand(1, 3, 256, 256))
    assert torch.allclose(apply_lut(img, lut), img, atol=1e-5)


def test_monotonicity_penalty_catches_inverted_lut():
    good = identity_lut()[None]
    inverted = 1.0 - good
    assert lut_regularization(good)[1] == 0
    assert lut_regularization(inverted)[1] > 0


def test_enhancer_full_resolution_in_chunks(untrained_checkpoint, astronaut):
    lut = LUTEnhancer(untrained_checkpoint, device=torch.device("cpu"))
    out = lut(astronaut, rows_per_chunk=100)  # 512 rows -> 6 chunks
    assert out.shape == astronaut.shape
    assert np.abs(out - astronaut).max() < 1e-4


def test_lut_plugs_into_enhance(untrained_checkpoint, astronaut):
    lut = LUTEnhancer(untrained_checkpoint, device=torch.device("cpu"))
    out, _, params = enhance(astronaut, lut=lut)
    assert out.shape == astronaut.shape
    assert params.style == 1.0
    assert params.describe()[0] == "Expert color & tone (learned)"
    # The LUT owns global color/tone; the rule-based versions are switched off.
    assert (params.wb_r, params.gamma, params.saturation) == (1.0, 1.0, 1.0)
