import numpy as np
import pytest
import torch
from skimage import data

from photofix import enhance
from photofix.degradations import gaussian_noise
from photofix.restore import NAFNet, Restorer


@pytest.fixture(scope="module")
def astronaut():
    return data.astronaut().astype(np.float32) / 255.0


def _save(model, tmp_path):
    path = tmp_path / "restorer.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, path)
    return path


@pytest.fixture(scope="module")
def perturbed_checkpoint(tmp_path_factory):
    """A model whose output actually depends on its input, so tiling errors would show up."""
    torch.manual_seed(0)
    model = NAFNet(width=8, enc_blocks=(1, 1), middle_blocks=1, dec_blocks=(1, 1))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return _save(model, tmp_path_factory.mktemp("restore"))


def test_untrained_model_is_identity():
    x = torch.rand(1, 3, 50, 70)  # not a multiple of 16: exercises padding
    assert torch.allclose(NAFNet()(x), x, atol=1e-6)


def test_tiled_matches_whole_image(perturbed_checkpoint, astronaut):
    restorer = Restorer(perturbed_checkpoint, device=torch.device("cpu"))
    whole = restorer(astronaut, tile=1024)  # 512² fits in one tile
    tiled = restorer(astronaut, tile=192, overlap=32)  # 9 tiles with feathered seams
    assert tiled.shape == astronaut.shape
    # Small differences come from zero-padding at tile borders; no seams means they stay tiny.
    assert np.abs(tiled - whole).mean() < 2e-3


def test_restorer_only_runs_on_flagged_photos(perturbed_checkpoint, astronaut):
    restorer = Restorer(perturbed_checkpoint, device=torch.device("cpu"))
    _, _, clean_params = enhance(astronaut, restorer=restorer)
    assert clean_params.restore == 0.0

    noisy = gaussian_noise(astronaut, 0.06, np.random.default_rng(0))
    _, analysis, params = enhance(noisy, restorer=restorer)
    assert analysis.scores["noise"] >= 0.3
    assert params.restore == 1.0
    assert params.denoise_h == 0.0  # NL-means is replaced, not stacked
    assert params.describe()[0] == "Neural denoise & deblur"
