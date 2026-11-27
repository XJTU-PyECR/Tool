"""Immutable paired/planning artifacts shared by GUI, project and reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping

from pyecr.services.experiment_design import PlanningSnapshot
from pyecr.services.paired_ecr import PairMember, PairedComparison
from pyecr.model_display import display_label


def _json(value: object) -> str:
    # Preserve the existing FitResult convention for unavailable legacy metrics.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=True)


@dataclass(frozen=True, slots=True)
class PairDesignState:
    member_a: PairMember | None = None
    member_b: PairMember | None = None
    comparison: PairedComparison | None = None
    plans: tuple[PlanningSnapshot, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plans", tuple(self.plans))
        if any(not isinstance(plan, PlanningSnapshot) for plan in self.plans):
            raise TypeError("plans must be immutable PlanningSnapshot values")
        if self.comparison is not None:
            if self.member_a is None or self.member_b is None:
                raise ValueError("Comparison requires both independent members")
            if (_json(self.comparison.member_a.to_dict()) != _json(self.member_a.to_dict())
                    or _json(self.comparison.member_b.to_dict()) != _json(self.member_b.to_dict())):
                raise ValueError("Comparison does not belong to the captured pair")

    @property
    def empty(self) -> bool:
        return self.member_a is None and self.member_b is None and not self.plans

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1,
                "member_a": None if self.member_a is None else self.member_a.to_dict(),
                "member_b": None if self.member_b is None else self.member_b.to_dict(),
                "comparison": None if self.comparison is None else self.comparison.to_dict(),
                "plans": [plan.to_dict() for plan in self.plans]}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PairDesignState:
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported pair/design state schema")
        return cls(None if value.get("member_a") is None else PairMember.from_dict(value["member_a"]),
                   None if value.get("member_b") is None else PairMember.from_dict(value["member_b"]),
                   None if value.get("comparison") is None else PairedComparison.from_dict(value["comparison"]),
                   tuple(PlanningSnapshot.from_dict(plan) for plan in value.get("plans", ())))

    def save(self, destination: str | Path) -> Path:
        """Save a new standalone planning/pair snapshot with an integrity digest."""
        path = Path(destination).resolve()
        payload = self.to_dict()
        envelope = {"kind": "pyecr_pair_design", "schema_version": 1, "payload": payload,
                    "sha256": hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
        return path

    @classmethod
    def open(cls, source: str | Path) -> PairDesignState:
        envelope = json.loads(Path(source).read_text(encoding="utf-8"))
        if envelope.get("kind") != "pyecr_pair_design" or envelope.get("schema_version") != 1:
            raise ValueError("Not a supported PyECR pair/design snapshot")
        payload = envelope["payload"]
        digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
        if digest != envelope.get("sha256"):
            raise ValueError("Pair/design snapshot SHA256 mismatch")
        return cls.from_dict(payload)

    def report_html(self) -> str:
        """Render the saved result; no forward, fit or sensitivity evaluation."""
        parts = ["<section><h2>Paired ECR Comparison and Experiment Design</h2>"]
        if self.comparison is not None:
            result = self.comparison
            parts.append(f"<h3>Independent pair: {escape(result.kind)}</h3>")
            parts.append("<p>Comparison ready: " + ("YES" if result.ready else "NO") + "</p>")
            for title, items in (("Blocking reasons", result.compatibility.blocking_reasons),
                                 ("Warnings", result.compatibility.warnings), ("Compatibility checks", result.compatibility.checks)):
                parts.append(f"<h4>{title}</h4><ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>")
            for member in (result.member_a, result.member_b):
                parts.append(f"<p>{escape(member.label)}: direction {escape(member.direction)}; "
                             f"FitResult ID <code>{member.fit_id}</code>; dataset SHA256 <code>{member.snapshot.dataset_sha256}</code>; "
                             f"model {escape(display_label(member.snapshot.analysis_config.model_id))}</p>")
            if result.ready:
                parts.append("<table><tr><th>Parameter</th><th>A</th><th>B</th><th>B − A</th><th>Relative</th><th>Symmetric relative</th><th>Unit</th></tr>")
                for row in result.parameters:
                    parts.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in (
                        result.parameter_label(row.parameter), row.value_a, row.value_b, row.difference_b_minus_a, row.relative_difference,
                        row.symmetric_relative_difference, row.unit)) + "</tr>")
                parts.append("</table><h4>Residual comparison / fit quality / identifiability</h4>")
                for row in result.curves:
                    parts.append(f"<h4>{escape(row.label)}</h4><p>SSE={row.SSE_reference:.6g}; RMSE={row.RMSE_reference:.6g}; "
                                 f"residual mean={row.residual_mean_reference:.6g}; SD={row.residual_sd_reference:.6g}; "
                                 f"max |residual|={row.residual_max_abs_reference:.6g}; unit={escape(row.residual_unit)}; "
                                 f"R²={row.R2:.6g}; Fit ready={row.fit_ready}; reliability={escape(row.reliability)}.</p>"
                                 f"<p>Identifiability: {escape(row.identifiability)} — {escape(row.identifiability_explanation)}</p>"
                                 f"<p>Model adequacy: {escape(row.model_adequacy)} — {escape(row.model_adequacy_explanation)}</p>")
                early = result.early_time_mismatch
                parts.append(f"<h4>Early-time mismatch</h4><p>Available={early.available}; fit RMS={early.rms_fit_difference}; "
                             f"max |fit difference|={early.max_abs_fit_difference}; observed RMS={early.rms_observed_difference}. "
                             f"{escape(early.reason)}</p>")
            parts.append("<ul>" + "".join(f"<li>{escape(note)}</li>" for note in result.interpretation_notes) + "</ul>")
        for index, plan in enumerate(self.plans, 1):
            parts.append(f"<h3>Planning snapshot {index} — 03 M2_SINGLE_DK</h3><p>Suggested region: "
                         + escape(", ".join(plan.recommended_candidate_ids) or "None under sampled assumptions") + "</p>")
            for region in plan.recommended_regions:
                parts.append(f"<p>Suggested full dimension intervals (cm): {escape(str(region.full_dimension_ranges_cm))}; "
                             f"active dimensions: {escape(str(region.active_dimensions))}; duration interval (s): {region.duration_range_s}. "
                             f"{escape(region.explanation)}</p>")
            parts.append("<details><summary>Planning inputs</summary><pre>" + escape(json.dumps(plan.request.to_dict(), indent=2, ensure_ascii=False)) + "</pre></details>")
            parts.append("<table><tr><th>Candidate</th><th>Full dimensions (cm) / active axes</th><th>Duration (s)</th>"
                         "<th>Bi ranges</th><th>Peak S_logD range</th><th>Peak S_logk range</th><th>Separation fraction</th><th>Region / reasons</th></tr>")
            for c in plan.candidates:
                values = (c.candidate_id, f"{c.geometry.full_dims_cm} / {c.geometry.active_dimensions}", c.duration_s,
                          c.bi_ranges, c.D_peak_sensitivity_range, c.k_peak_sensitivity_range,
                          c.supported_fraction, f"recommended={c.recommended_region}; " + "; ".join(c.reasons))
                parts.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>")
            parts.append("</table><ul>" + "".join(f"<li>{escape(note)}</li>" for note in plan.notes) + "</ul>")
        parts.append("</section>")
        return "\n".join(parts)

    def export_report(self, destination: str | Path) -> Path:
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write('<!doctype html><html><head><meta charset="utf-8"><title>PyECR Pair / Design</title>'
                         '<style>body{font:15px sans-serif;margin:32px;max-width:1300px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px}td{vertical-align:top}code{overflow-wrap:anywhere}</style></head><body>'
                         + self.report_html() + "</body></html>\n")
        return path
