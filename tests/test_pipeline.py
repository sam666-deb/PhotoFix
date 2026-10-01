import numpy as np
import pytest
from skimage import data

from photofix import EditParams, analyze, apply_edits, enhance
from photofix.analyzer import DEFECTS
from photofix.degradations import exposure, gaussian_noise, random_degrade
from photofix.imageio import encode_jpeg, load_image
from photofix.metrics import delta_e, psnr


@pytest.fixture(scope="module")
def astronaut():
    return data.astronaut().astype(np.float32) / 255.0


def test_identity_params_do_nothing(astronaut):
    assert np.array_equal(apply_edits(astronaut, EditParams()), astronaut)


def test_clean_photo_is_left_alone(astronaut):
    out, analysis, params = enhance(astronaut)
    assert analysis.defects() == []
    assert delta_e(out, astronaut) < 1.0


def test_detects_and_fixes_underexposure(astronaut):
    dark = exposure(astronaut, -2.5)
    out, analysis, params = enhance(dark)
    assert analysis.scores["underexposure"] >= 0.5
    assert params.gamma < 1.0
    assert np.median(out) > np.median(dark)
    assert psnr(out, astronaut) > psnr(dark, astronaut)


def test_detects_and_reduces_noise(astronaut):
    noisy = gaussian_noise(astronaut, 0.05, np.random.default_rng(0))
    out, analysis, params = enhance(noisy)
    assert analysis.scores["noise"] >= 0.5
    assert params.denoise_h > 0
    assert psnr(out, astronaut) > psnr(noisy, astronaut)


def test_strength_zero_returns_original(astronaut):
    dark = exposure(astronaut, -2.0)
    out, _, _ = enhance(dark, strength=0.0)
    assert np.allclose(out, dark)


def test_random_degrade_labels(astronaut):
    rng = np.random.default_rng(1)
    for _ in range(5):
        out, labels = random_degrade(astronaut, rng)
        assert out.shape == astronaut.shape and out.dtype == np.float32
        assert set(labels) == set(DEFECTS)
        assert 1 <= sum(v > 0 for v in labels.values()) <= 3
        assert not (labels["underexposure"] and labels["overexposure"])


def test_analysis_scores_in_range(astronaut):
    scores = analyze(exposure(astronaut, 2.0)).scores
    assert set(scores) == set(DEFECTS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_jpeg_roundtrip(astronaut):
    rgb, _ = load_image(encode_jpeg(astronaut))
    assert rgb.shape == astronaut.shape
    assert psnr(rgb, astronaut) > 35


def test_shadow_lift_does_not_tint_near_black():
    # A near-black pixel with a faint blue bias used to be scaled ~50x into saturated blue.
    img = np.full((64, 64, 3), [0.004, 0.006, 0.02], dtype=np.float32)
    out = apply_edits(img, EditParams(shadow_lift=0.6, gamma=0.8))
    r, g, b = out[0, 0]
    assert out.mean() > img.mean()  # it was lifted
    assert b - r < 0.05  # ...but stayed close to neutral


def _analysis(**scores):
    from photofix.analyzer import Analysis

    return Analysis(scores={d: scores.get(d, 0.0) for d in DEFECTS}, stats={"median": 0.45, "p99": 0.9})


def test_finish_adds_contrast_only_when_flat(astronaut):
    from photofix.enhancer import plan_finish

    flat = 0.4 + 0.2 * astronaut  # washed-out version
    params = EditParams(shadow_lift=0.5)
    assert plan_finish(flat, params, _analysis()).contrast > 0.2
    assert plan_finish(astronaut, params, _analysis()).contrast < 0.1


def test_finish_backs_off_sharpening_on_noisy_photos(astronaut):
    from photofix.enhancer import plan_finish

    params = EditParams(shadow_lift=0.5)
    assert plan_finish(astronaut, params, _analysis()).sharpen_amount > 0
    assert plan_finish(astronaut, params, _analysis(noise=0.8)).sharpen_amount == 0


def test_shadow_lift_keeps_local_contrast(astronaut):
    # Lifting shadows should brighten dark areas without flattening their texture.
    dark = astronaut * 0.35
    out, _, params = enhance(dark)
    assert params.clarity > 0
    assert np.median(out) > np.median(dark)
    assert out.std() > dark.std()
