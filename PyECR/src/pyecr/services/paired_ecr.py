"""Independent ECR comparisons. No raw curves are combined or refitted here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite
from typing import Mapping

import numpy as np

from pyecr.models.ids import ModelID
from pyecr.preprocessing.quality import evaluate_fit_quality
from pyecr.report.snapshot import AnalysisSnapshot
from pyecr.services.guided_refit import fit_result_id
from pyecr.units import factor_to_reference, validate_pair
from pyecr.config import _freeze_json, _thaw_json


PAIR_KINDS = ("oxidation_reduction", "hydration_dehydration", "generic")
DIRECTIONS = ("oxidation", "reduction", "hydration", "dehydration", "increase", "decrease", "custom")
_KINETICS = {
    ModelID.M0_SINGLE_D: ("D_cm2_s",),
    ModelID.M1_SINGLE_DK: ("D_cm2_s", "k_cm_s"),
    ModelID.M2_DUAL_HO_D: ("D_H_cm2_s", "D_O_cm2_s"),
    ModelID.M3_DUAL_HO_DK: ("D_H_cm2_s", "k_H_cm_s", "D_O_cm2_s", "k_O_cm_s"),
    ModelID.M7_SINGLE_K: ("k_cm_s",),
    ModelID.M8_DUAL_K: ("k_1_cm_s", "k_2_cm_s"),
}


def point_comparison_supported(snapshot: AnalysisSnapshot) -> bool:
    """Descriptive point comparisons do not require calibrated interval coverage.

    Current Diagnostics separates these claims. Legacy snapshots still need
    their original reliability flag; incomplete or conflicting new evidence is
    never treated as permission to compare.
    """
    diagnostics = snapshot.diagnostics_result
    if snapshot.fit_result.exit_status <= 0:
        return False
    evidence = diagnostics.get("evaluation")
    if evidence is None:
        return diagnostics.get("reliable") is True
    if not isinstance(evidence, Mapping):
        return False
    if (evidence.get("policy_version") != "diagnostics-evaluation/2"
            or evidence.get("point_estimates_usable") is not True
            or evidence.get("blocking_reasons", ("missing",))
            or evidence.get("adequacy_status") not in {"supported", "caution"}
            or evidence.get("identifiability_status") not in {"supported", "caution"}
            or evidence.get("interval_status") not in {"conditional_iid", "assumptions_unverified"}):
        return False
    parameters=evidence.get("parameters", ())
    if not isinstance(parameters,(tuple,list)):
        return False
    rows = {p.get("name"): p for p in parameters if isinstance(p, Mapping)}
    return all(name in rows and rows[name].get("precision") == "supported"
               and rows[name].get("boundary") in {"clear", "unconfigured"}
               for name in _KINETICS.get(snapshot.analysis_config.model_id, ()))


@dataclass(frozen=True, slots=True)
class PairMember:
    label: str
    direction: str
    snapshot: AnalysisSnapshot
    context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip() or self.direction not in DIRECTIONS:
            raise ValueError("Pair member needs a label and an explicit supported step direction")
        if not isinstance(self.snapshot, AnalysisSnapshot):
            raise TypeError("Pair member requires an immutable AnalysisSnapshot")
        if set(self.context) - {"temperature_k", "initial_condition", "final_condition", "step_direction"}:
            raise ValueError("Pair context accepts temperature_k, initial_condition, final_condition and step_direction")
        object.__setattr__(self, "context", _freeze_json(self.context, path="pair_context"))

    @property
    def fit_id(self) -> str:
        return fit_result_id(self.snapshot.fit_result)

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "direction": self.direction,
                "fit_id": self.fit_id, "snapshot": self.snapshot.to_dict(), "context": _thaw_json(self.context)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PairMember:
        def plain(item: object) -> object:
            if isinstance(item, Mapping):
                return {key: plain(v) for key, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [plain(v) for v in item]
            return item
        snapshot = plain(value["snapshot"])
        for name in ("SSE", "RMSE", "R2", "AIC", "AICc", "BIC"):
            snapshot["fit_result"][name] = float(snapshot["fit_result"][name])
        for candidate in snapshot["fit_result"].get("candidates", ()):
            candidate["SSE"] = float(candidate["SSE"])
        member = cls(str(value["label"]), str(value["direction"]),
                     AnalysisSnapshot.from_dict(snapshot), value.get("context", {}))
        if value.get("fit_id") != member.fit_id:
            raise ValueError("Pair member FitResult ID mismatch")
        return member


@dataclass(frozen=True, slots=True)
class PairCompatibility:
    compatible: bool
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParameterDifference:
    parameter: str
    value_a: float
    value_b: float
    difference_b_minus_a: float
    relative_difference: float
    symmetric_relative_difference: float
    unit: str


@dataclass(frozen=True, slots=True)
class CurveComparisonSummary:
    label: str
    fit_id: str
    SSE_reference: float
    RMSE_reference: float
    residual_mean_reference: float
    residual_sd_reference: float
    residual_max_abs_reference: float
    R2: float
    fit_ready: bool
    quality_warnings: tuple[str, ...]
    reliability: str
    identifiability: str
    identifiability_explanation: str
    model_adequacy: str
    model_adequacy_explanation: str
    residual_unit: str


@dataclass(frozen=True, slots=True)
class EarlyTimeMismatch:
    available: bool
    time_s: tuple[float, ...] = ()
    fit_progress_difference_b_minus_a: tuple[float, ...] = ()
    observed_progress_difference_b_minus_a: tuple[float, ...] = ()
    rms_fit_difference: float | None = None
    max_abs_fit_difference: float | None = None
    rms_observed_difference: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PairedComparison:
    kind: str
    member_a: PairMember
    member_b: PairMember
    compatibility: PairCompatibility
    parameters: tuple[ParameterDifference, ...] = ()
    curves: tuple[CurveComparisonSummary, ...] = ()
    early_time_mismatch: EarlyTimeMismatch = EarlyTimeMismatch(False)
    interpretation_notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.compatibility.compatible

    def parameter_label(self, parameter: str) -> str:
        """Describe the saved slots without assigning chemical identities."""
        if self.member_a.snapshot.analysis_config.model_id is ModelID.M8_DUAL_K:
            labels = self.member_a.snapshot.fit_result.diagnostics["dual_surface_k"]["component_labels"]
            if parameter in _KINETICS[ModelID.M8_DUAL_K]:
                return f"{parameter} — {labels[_KINETICS[ModelID.M8_DUAL_K].index(parameter)]}"
        return parameter

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "kind": self.kind,
                "member_a": self.member_a.to_dict(), "member_b": self.member_b.to_dict(),
                "compatibility": asdict(self.compatibility),
                "parameters": [asdict(row) for row in self.parameters],
                "curves": [asdict(row) for row in self.curves],
                "early_time_mismatch": asdict(self.early_time_mismatch),
                "interpretation_notes": list(self.interpretation_notes)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PairedComparison:
        if value.get("schema_version") != 1 or value.get("kind") not in PAIR_KINDS:
            raise ValueError("Unsupported paired comparison snapshot")
        check = dict(value["compatibility"])
        for key in ("blocking_reasons", "warnings", "checks"):
            check[key] = tuple(check.get(key, ()))
        early = dict(value["early_time_mismatch"])
        for key in ("time_s", "fit_progress_difference_b_minus_a", "observed_progress_difference_b_minus_a"):
            early[key] = tuple(early.get(key, ()))
        curves = []
        for row in value.get("curves", ()):
            payload = dict(row)
            payload["quality_warnings"] = tuple(payload["quality_warnings"])
            curves.append(CurveComparisonSummary(**payload))
        return cls(str(value["kind"]), PairMember.from_dict(value["member_a"]),
                   PairMember.from_dict(value["member_b"]), PairCompatibility(**check),
                   tuple(ParameterDifference(**row) for row in value.get("parameters", ())),
                   tuple(curves), EarlyTimeMismatch(**early),
                   tuple(value.get("interpretation_notes", ())))


def _metadata(member: PairMember, blockers: list[str]) -> Mapping[str, object]:
    metadata = dict(member.snapshot.dataset.metadata)
    for key, value in member.context.items():
        if metadata.get(key) is not None and metadata[key] != value:
            blockers.append(f"{member.label}: pair context contradicts recorded {key} metadata")
        else:
            metadata[key] = value
    return metadata


def _direction_evidence(member: PairMember, metadata: Mapping[str, object], blockers: list[str]) -> None:
    snapshot = member.snapshot
    for key in ("step_direction", "step_type"):
        recorded = metadata.get(key)
        if recorded is not None and str(recorded).strip().lower() != member.direction:
            blockers.append(f"{member.label}: declared direction disagrees with {key} metadata")
    if snapshot.analysis_config.model_id in {ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK, ModelID.M8_DUAL_K}:
        if snapshot.analysis_config.model.step_type != member.direction:
            blockers.append(f"{member.label}: direction disagrees with the fitted dual-process step definition")
    if member.direction in {"increase", "decrease"}:
        raw = snapshot.dataset.raw_signal
        edge = max(1, min(5, raw.size // 4))
        delta = float(np.median(raw[-edge:]) - np.median(raw[:edge]))
        if delta == 0 or (delta > 0) != (member.direction == "increase"):
            blockers.append(f"{member.label}: declared signal direction disagrees with raw observations")


def _unit_scale(snapshot: AnalysisSnapshot) -> tuple[float, str]:
    metadata = snapshot.dataset.metadata
    kind = str(metadata.get("signal_type", "unknown"))
    unit = str(metadata.get("signal_unit", ""))
    key = validate_pair(kind, unit)
    if key in {"custom", "arbitrary"}:
        return 1.0, unit
    factor = factor_to_reference(unit)
    label = {"normalized": "1", "resistance": "Ω", "conductance": "S", "conductivity": "S/cm"}[kind]
    return factor, label


def _dual_k_comparability(a: PairMember, b: PairMember, blockers: list[str],
                          warnings: list[str], checks: list[str]) -> None:
    """Only compare the already reported labels; never remap or refit slots."""
    records = []
    for member in (a, b):
        snapshot = member.snapshot
        record = snapshot.fit_result.diagnostics.get("dual_surface_k", {})
        options = snapshot.analysis_config.model_specific.get("dual_surface_k", {})
        semantics = record.get("label_semantics")
        labels = record.get("component_labels", ())
        labels = tuple(labels) if isinstance(labels, (tuple, list)) else ()
        if (semantics not in {"effective_modes", "attributed_species"}
                or semantics != options.get("label_semantics", "effective_modes")
                or len(labels) != 2 or not all(isinstance(v, str) and v.strip() for v in labels)):
            blockers.append(f"{member.label}: dual-k label semantics or captured labels are missing/conflicting")
        if semantics == "attributed_species":
            mapping = record.get("label_mapping", {})
            evidence = record.get("label_evidence")
            if (not isinstance(mapping, Mapping) or set(mapping) != {"1", "2"}
                    or labels != tuple(mapping.get(str(i)) for i in (1, 2))
                    or len(set(labels)) != 2 or not isinstance(evidence, str) or not evidence.strip()
                    or mapping != options.get("label_mapping", {}) or evidence != options.get("label_evidence")):
                blockers.append(f"{member.label}: attributed dual-k comparison requires matching saved mappings and evidence")
        elif semantics == "effective_modes":
            params = snapshot.fit_result.params
            if (tuple(labels) != ("Fast effective process", "Slow effective process")
                    or params.get("k_1_cm_s", 0) < params.get("k_2_cm_s", 0)):
                blockers.append(f"{member.label}: dual-k fast/slow reported ordering is inconsistent")
        if any(record.get("parameter_roles", {}).get(name) != "free" for name in _KINETICS[ModelID.M8_DUAL_K]):
            blockers.append(f"{member.label}: both dual-k estimates must be independently fitted, not fixed")
        records.append(record)
    ra, rb = records
    if ra.get("label_semantics") != rb.get("label_semantics"):
        blockers.append("Dual-k label semantics differ; effective modes and attributed species cannot be mixed")
    elif ra.get("label_semantics") == "attributed_species":
        if ra.get("label_mapping") != rb.get("label_mapping"):
            blockers.append("Dual-k attributed label mappings differ; align explicit source labels before comparison")
        elif len(ra.get("component_labels", ())) == 2:
            checks.append(f"Dual-k attributed slots: k_1={ra['component_labels'][0]}, k_2={ra['component_labels'][1]}; each source retains its own evidence")
    else:
        warnings.append("EFFECTIVE_MODE_PAIR: k_1/k_2 compare fast/slow effective modes in each fit. Rate ordering can exchange physical processes; these are not automatically H/O or species-matched differences.")
    if any(ra.get("parameter_roles", {}).get(n) != rb.get("parameter_roles", {}).get(n) for n in ("A_1", "A_2", "y0")):
        blockers.append("Dual-k free/fixed amplitude or baseline policies differ")
    checks.append("Analytic dual-k comparison is conditional on captured geometry, signs and fixed inputs; no D or Bi is inferred")


def check_pair_compatibility(a: PairMember, b: PairMember, kind: str) -> PairCompatibility:
    if kind not in PAIR_KINDS:
        raise ValueError("Unknown pair kind")
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    expected = {"oxidation_reduction": {"oxidation", "reduction"},
                "hydration_dehydration": {"hydration", "dehydration"}}
    if kind in expected and {a.direction, b.direction} != expected[kind]:
        blockers.append(f"{kind} requires complementary step directions")
    sa, sb = a.snapshot, b.snapshot
    if "fixed_d" in sa.analysis_config.model_specific or "fixed_d" in sb.analysis_config.model_specific:
        return PairCompatibility(False,("Conditional fixed-D Pair comparison is unsupported",),(),())
    if any(s.analysis_config.model_id not in _KINETICS for s in (sa, sb)):
        return PairCompatibility(False, ("Selected model does not support Pair comparison",), (), ())
    ma, mb = _metadata(a, blockers), _metadata(b, blockers)
    if sa.dataset_sha256 == sb.dataset_sha256:
        blockers.append("Pair requires two independent datasets, not the same captured curve")
    for member, metadata in ((a, ma), (b, mb)):
        _direction_evidence(member, metadata, blockers)
        s = member.snapshot
        quality = evaluate_fit_quality(s.dataset, s.analysis_config)
        blockers.extend(f"{member.label}: {reason}" for reason in quality.blocking_reasons)
        diagnostics = s.diagnostics_result
        dispatch = s.fit_result.diagnostics.get("analysis_dispatch", {})
        if isinstance(dispatch, Mapping) and dispatch.get("selected_model_id", s.analysis_config.model_id.value) != s.analysis_config.model_id.value:
            blockers.append(f"{member.label}: final selected model differs from the captured requested model")
        if diagnostics.get("model_id") != s.analysis_config.model_id.value:
            blockers.append(f"{member.label}: current Diagnostics for the selected model are required")
        if (tuple(diagnostics.get("time_s", ())) != tuple(s.dataset.processed_time)
                or tuple(diagnostics.get("residuals", ())) != tuple(s.fit_result.residuals)):
            blockers.append(f"{member.label}: Diagnostics time/residual arrays do not match this fit")
        if not point_comparison_supported(s):
            blockers.append(f"{member.label}: fit reliability check failed")
        elif diagnostics.get("reliable") is not True:
            warnings.append(f"{member.label}: point estimates support descriptive comparison; local interval error assumptions "
                            "are unverified. Differences have no calibrated confidence interval or significance test.")
        advisor = diagnostics.get("reliability_advisor", {})
        if isinstance(advisor, Mapping):
            if not advisor:
                warnings.append(f"{member.label}: legacy Diagnostics lacks parameter / adequacy evidence")
            if advisor.get("identifiability_status") in {"weak", "unavailable"}:
                blockers.append(f"{member.label}: parameter identifiability is weak or unavailable")
            if advisor.get("model_adequacy_status") in {"poor", "unavailable"}:
                blockers.append(f"{member.label}: model adequacy is poor or unavailable")
            if "caution" in (advisor.get("identifiability_status"), advisor.get("model_adequacy_status")):
                warnings.append(f"{member.label}: Diagnostics advises caution; inspect captured evidence")
        else:
            warnings.append(f"{member.label}: legacy Diagnostics lacks parameter / adequacy evidence")
        parameter_rows = {row.get("name"): row for row in diagnostics.get("parameters", ()) if isinstance(row, Mapping)}
        for name in _KINETICS[s.analysis_config.model_id]:
            if name not in parameter_rows or parameter_rows[name].get("reliable") is not True:
                blockers.append(f"{member.label}: {name} parameter reliability is not supported")
    if sa.analysis_config.model_id != sb.analysis_config.model_id:
        blockers.append("Selected models differ")
    else:
        checks.append("Selected models match")
    if sa.analysis_config.model_id is sb.analysis_config.model_id is ModelID.M8_DUAL_K:
        _dual_k_comparability(a, b, blockers, warnings, checks)
    if sa.analysis_config.model_id is sb.analysis_config.model_id is ModelID.M7_SINGLE_K:
        ra, rb = sa.fit_result.diagnostics["surface_k"], sb.fit_result.diagnostics["surface_k"]
        if ra["observation_mode"] != rb["observation_mode"] or any(
                ra["parameter_roles"][name] != rb["parameter_roles"][name]
                for name in ("amplitude", "baseline")):
            blockers.append("Single-k observation modes or free/fixed amplitude/baseline policies differ")
        checks.append("Analytic single-k comparison; each estimate is conditional on its captured fixed inputs; no D or Bi is inferred")
    ga, gb = sa.analysis_config.model.geometry, sb.analysis_config.model.geometry
    if ga.active_dimensions != gb.active_dimensions or not np.allclose(ga.full_dims_cm, gb.full_dims_cm, rtol=1e-6, atol=0.0):
        blockers.append("Full geometry or active dimensions differ")
    else:
        checks.append("Full geometry and active dimensions match")
    for prefix in ("", "raw_"):
        ta, tb = ma.get(prefix + "signal_type"), mb.get(prefix + "signal_type")
        if ta != tb:
            blockers.append(f"{prefix or 'processed_'}signal types differ")
        try:
            ka = validate_pair(str(ta), str(ma.get(prefix + "signal_unit", "")))
            kb = validate_pair(str(tb), str(mb.get(prefix + "signal_unit", "")))
            if ka in {"custom", "arbitrary"} or kb in {"custom", "arbitrary"}:
                if ka != kb or ma.get(prefix + "signal_unit") != mb.get(prefix + "signal_unit"):
                    blockers.append(f"{prefix or 'processed_'}signal units are not comparable")
            elif ka != kb:
                warnings.append(f"{prefix or 'processed_'}signal unit scales differ; residuals use canonical reference units")
        except ValueError:
            blockers.append(f"Invalid or unknown {prefix or 'processed_'}signal type/unit")
    temperatures = (ma.get("temperature_k"), mb.get("temperature_k"))
    if all(t is not None for t in temperatures):
        try:
            valid = all(not isinstance(t, bool) and isfinite(float(t)) and float(t) > 0 for t in temperatures)
            if not valid or not isclose(float(temperatures[0]), float(temperatures[1]), rel_tol=1e-6, abs_tol=0.1):
                blockers.append("Temperature metadata differ or are invalid (canonical kelvin required)")
        except (TypeError, ValueError):
            blockers.append("Temperature metadata must be numeric kelvin")
    else:
        warnings.append("Temperature metadata missing on one or both curves")
    finals = (ma.get("final_condition"), mb.get("final_condition"))
    if all(c is not None for c in finals):
        if finals[0] != finals[1]:
            reciprocal = (ma.get("initial_condition") == finals[1] and mb.get("initial_condition") == finals[0])
            if not reciprocal:
                blockers.append("Final conditions differ without reciprocal initial/final condition metadata")
            else:
                checks.append("Initial/final conditions form a reciprocal pair")
        else:
            checks.append("Final conditions match")
    else:
        warnings.append("Final-condition metadata missing on one or both curves")
    if sa.analysis_config.boundary.mode != sb.analysis_config.boundary.mode:
        blockers.append("Boundary assumptions differ")
    elif sa.analysis_config.boundary != sb.analysis_config.boundary:
        warnings.append("Boundary histories or flush constants differ; both independent fits retain their own correction")
    if sa.analysis_config.model_id == sb.analysis_config.model_id:
        for name in _KINETICS[sa.analysis_config.model_id]:
            if name not in sa.fit_result.params or name not in sb.fit_result.params:
                blockers.append(f"{name}: both estimates must be independently fitted, not fixed")
    checks.append(f"Directions reviewed: {a.direction} / {b.direction}; no chemistry-to-signal-sign assumption")
    return PairCompatibility(not blockers, tuple(dict.fromkeys(blockers)), tuple(dict.fromkeys(warnings)), tuple(checks))


def _curve_summary(member: PairMember) -> CurveComparisonSummary:
    s = member.snapshot
    factor, unit = _unit_scale(s)
    r = np.asarray(s.fit_result.residuals) * factor
    quality = evaluate_fit_quality(s.dataset, s.analysis_config)
    advisor = s.diagnostics_result.get("reliability_advisor", {})
    advisor = advisor if isinstance(advisor, Mapping) else {}
    return CurveComparisonSummary(
        member.label, member.fit_id, s.fit_result.SSE * factor**2, s.fit_result.RMSE * factor,
        float(np.mean(r)), float(np.std(r, ddof=1)), float(np.max(np.abs(r))), s.fit_result.R2,
        quality.ready, quality.warnings, str(s.diagnostics_result.get("reliability_label", "unknown")),
        str(advisor.get("identifiability_status", "legacy unavailable")),
        str(advisor.get("identifiability_explanation", "No captured P50 advisor")),
        str(advisor.get("model_adequacy_status", "legacy unavailable")),
        str(advisor.get("model_adequacy_explanation", "No captured P50 advisor")), unit)


def _early_mismatch(a: AnalysisSnapshot, b: AnalysisSnapshot) -> EarlyTimeMismatch:
    ta, tb = a.dataset.processed_time, b.dataset.processed_time
    start, end = max(float(ta[0]), float(tb[0])), min(float(ta[-1]), float(tb[-1]))
    if end <= start:
        return EarlyTimeMismatch(False, reason="No common fit-time interval; extrapolation is forbidden")
    cutoff = start + 0.2 * (end - start)
    grid = np.unique(np.concatenate((ta[(ta >= start) & (ta <= cutoff)], tb[(tb >= start) & (tb <= cutoff)])))
    if grid.size < 3:
        return EarlyTimeMismatch(False, reason="Fewer than three observed early times in the common interval")

    def progress(s: AnalysisSnapshot) -> tuple[np.ndarray, np.ndarray]:
        signal = s.dataset.processed_signal
        if s.dataset.metadata.get("signal_type") == "normalized":
            initial, amplitude = 0.0, 1.0
        else:
            edge = max(1, min(5, signal.size // 4))
            initial = float(np.median(signal[:edge]))
            amplitude = float(np.median(signal[-edge:]) - initial)
        if abs(amplitude) < np.finfo(float).tiny:
            raise ValueError("Early mismatch requires nonzero response amplitude")
        return (s.fit_result.y_fit - initial) / amplitude, (signal - initial) / amplitude

    fa, oa = progress(a)
    fb, ob = progress(b)
    fd = np.interp(grid, tb, fb) - np.interp(grid, ta, fa)
    od = np.interp(grid, tb, ob) - np.interp(grid, ta, oa)
    return EarlyTimeMismatch(True, tuple(grid), tuple(fd), tuple(od), float(np.sqrt(np.mean(fd**2))),
                             float(np.max(np.abs(fd))), float(np.sqrt(np.mean(od**2))),
                             "First 20% of the common elapsed fit-time span; linear interpolation inside coverage only. "
                             "Raw signals use their own edge levels for a direction-oriented shape diagnostic; no curves are averaged.")


def compare_paired_ecr(a: PairMember, b: PairMember, kind: str = "generic") -> PairedComparison:
    compatibility = check_pair_compatibility(a, b, kind)
    if not compatibility.compatible:
        return PairedComparison(kind, a, b, compatibility,
                                interpretation_notes=("Comparison blocked; resolve every compatibility reason before interpreting parameter differences.",))
    differences = []
    for name in _KINETICS[a.snapshot.analysis_config.model_id]:
        va, vb = a.snapshot.fit_result.params[name], b.snapshot.fit_result.params[name]
        if not (isfinite(va) and isfinite(vb) and va > 0 and vb > 0):
            raise ValueError("Compared kinetic parameters must be positive and finite")
        differences.append(ParameterDifference(name, va, vb, vb - va, (vb - va) / va,
                                                2 * (vb - va) / (abs(va) + abs(vb)),
                                                a.snapshot.analysis_config.parameter_units[name]))
    return PairedComparison(
        kind, a, b, compatibility, tuple(differences), (_curve_summary(a), _curve_summary(b)),
        _early_mismatch(a.snapshot, b.snapshot),
        ("Differences describe separately fitted responses under the recorded conditions; no specific chemical mechanism is inferred.",
         "Relative difference = (B − A) / A; symmetric relative difference = 2(B − A) / (|A| + |B|).",
         "Residual SSE depends on observation count; compare RMSE, residual structure, and time coverage alongside SSE.",
         "Parameter identifiability and model adequacy are distinct; differences are not a formal significance test.",
         "Missing condition metadata limits comparability; reciprocal conditions do not establish identical material states."))
