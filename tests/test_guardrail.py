import numpy as np
import pytest
from skimage import data

from photofix.degradations import exposure
from photofix.guardrail import LIMITS, guard, measure
from photofix.pipeline import Pipeline


@pytest.fixture(scope="module")
def astronaut():
    return data.astronaut().astype(np.float32) / 255.0


def test_identical_images_measure_clean(astronaut):
    m = measure(astronaut, astronaut)
    assert set(m) == set(LIMITS)
    assert all(m[k] <= 0 for k in ("highlights_clipped", "shadows_crushed", "oversaturated"))
    assert m["texture_amplified"] == pytest.approx(1.0)


def test_good_edit_passes_untouched(astronaut):
    dark = exposure(astronaut, -1.0)
    out, report = guard(dark, astronaut)  # brightening back to the original is a reasonable edit
    assert not report.adjusted
    assert out is astronaut
    assert report.describe() is None


def test_blown_highlights_get_toned_down(astronaut):
    blown = np.clip(astronaut * 1.8, 0, 1)
    out, report = guard(astronaut, blown)
    assert "highlights_clipped" in report.violations
    assert 0.0 <= report.strength < 1.0
    assert "highlights" in report.describe()
    # The returned image is a blend between original and edit, at the reported strength.
    np.testing.assert_allclose(out, np.clip(astronaut + (blown - astronaut) * report.strength, 0, 1), atol=1e-6)


def test_lightness_change_alone_is_not_a_skin_shift(astronaut):
    # a*b* only: brightening a face must not count as changing its color.
    m = measure(astronaut, exposure(astronaut, 0.5))
    assert m["skin_shift"] < LIMITS["skin_shift"][0]


def test_pipeline_reports_stages_and_skips_guard_for_untouched_photos(astronaut):
    pipeline = Pipeline()  # classical analyzer, no learned models
    result = pipeline.run(astronaut)
    assert result.params.is_identity()  # already-good photo
    assert "guardrail" not in result.timings and "analyze" in result.timings
    assert result.steps == []

    result = pipeline.run(exposure(astronaut, -2.5))
    assert {"analyze", "edits", "guardrail", "total"} <= set(result.timings)
    payload = result.to_dict()
    assert payload["style"] == "natural" and payload["guardrail"]["strength"] <= 1.0


def test_pipeline_rejects_unavailable_style(astronaut):
    with pytest.raises(ValueError):
        Pipeline().run(astronaut, style="pro")
