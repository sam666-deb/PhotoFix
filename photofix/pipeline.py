"""Phase 4: one orchestrated pipeline used by the web server, the gallery and the benchmarks.

    analyze ─► restore (if noisy/blurry) ─► style (Pro: learned LUT) ─► edits + finishing ─► guardrail

`Pipeline.load()` picks up whichever trained models exist in checkpoints/, so the app degrades
gracefully: no analyzer checkpoint -> classical analyzer; no LUT -> Natural style only; no restorer ->
classical denoise/sharpen. Every result says what ran, why, and how long each stage took.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from photofix.analyzer import Analysis
from photofix.enhancer import EditParams, enhance
from photofix.guardrail import GuardReport, guard

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass
class Result:
    image: np.ndarray
    analysis: Analysis
    params: EditParams
    guard: GuardReport
    style: str
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def steps(self) -> list[str]:
        steps = self.params.describe()
        note = self.guard.describe()
        return steps + [note] if note else steps

    def to_dict(self) -> dict:
        return {"analysis": self.analysis.to_dict(), "params": self.params.to_dict(), "steps": self.steps,
                "guardrail": self.guard.to_dict(), "style": self.style, "timings_ms": self.timings}


class Pipeline:
    def __init__(self, analyzer=None, lut=None, restorer=None, guardrail: bool = True):
        self.analyzer, self.lut, self.restorer, self.guardrail = analyzer, lut, restorer, guardrail

    @classmethod
    def load(cls, checkpoints: str | Path = CHECKPOINTS, analyzer: bool = True, lut: bool = True,
             restorer: bool = True, guardrail: bool = True) -> "Pipeline":
        """Load every enabled component whose checkpoint exists."""
        checkpoints = Path(checkpoints)
        components = {}
        if analyzer and (checkpoints / "analyzer.pt").exists():
            from photofix.net import DLAnalyzer

            components["analyzer"] = DLAnalyzer(checkpoints / "analyzer.pt")
        if lut and (checkpoints / "lut.pt").exists():
            from photofix.lut import LUTEnhancer

            components["lut"] = LUTEnhancer(checkpoints / "lut.pt")
        if restorer and (checkpoints / "restorer.pt").exists():
            from photofix.restore import Restorer

            components["restorer"] = Restorer(checkpoints / "restorer.pt")
        return cls(**components, guardrail=guardrail)

    @property
    def styles(self) -> tuple[str, ...]:
        # natural: rule-based corrections + finishing (vivid, suits already-processed phone photos)
        # pro:     global color/tone learned from a professional retoucher (MIT-Adobe FiveK, Expert C)
        return ("natural", "pro") if self.lut else ("natural",)

    def describe(self) -> dict:
        return {"analyzer": self.analyzer.name if self.analyzer else "classical",
                "restorer": self.restorer.name if self.restorer else "classical",
                "styles": list(self.styles), "guardrail": self.guardrail}

    def run(self, rgb: np.ndarray, style: str = "natural") -> Result:
        if style not in self.styles:
            raise ValueError(f"Unknown or unavailable style {style!r}; available: {', '.join(self.styles)}.")
        timings: dict[str, float] = {}
        out, analysis, params = enhance(
            rgb, net=self.analyzer, lut=self.lut if style == "pro" else None, restorer=self.restorer, timings=timings
        )
        report = GuardReport()
        if self.guardrail and not params.is_identity():
            start = time.perf_counter()
            out, report = guard(np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False), out)
            timings["guardrail"] = round((time.perf_counter() - start) * 1000, 1)
        timings["total"] = round(sum(timings.values()), 1)
        return Result(out, analysis, params, report, style, timings)
