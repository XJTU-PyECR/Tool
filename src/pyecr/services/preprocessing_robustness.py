"""Explicit, side-effect-free sensitivity to preprocessing choices (P65B)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite, log10, sqrt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.fitting.single import prepare_single_objective
from pyecr.fitting.dual_objective import DualHOObjectiveEvaluator, _context_for_level
from pyecr.fitting.dual_ho import _Parameterization
from pyecr.io.dataset import EcrDataset
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.preprocessing import PreprocessConfig, ProcessingStep
from pyecr.preprocessing.optional import preview_downsampling, preview_drift
from pyecr.preprocessing.quality import evaluate_fit_quality
from pyecr.report.snapshot import dataset_sha256
from pyecr.services.analysis import AnalysisService, CancellationToken, _dual_fit_adapter
from pyecr.services.diagnostics_service import DiagnosticsService
from pyecr.services.guided_refit import fit_result_id
from pyecr.services.workflow import AnalysisSession, WorkflowRevision

_KINETIC = ("D_cm2_s", "k_cm_s", "D_H_cm2_s", "k_H_cm_s", "D_O_cm2_s", "k_O_cm_s")
_KINDS = {"initial_window", "final_window", "t0", "drift_reference", "downsample"}
_MAX_SCENARIOS = 8


@dataclass(frozen=True, slots=True)
class RobustnessVariation:
    """One user-selected replacement or added downsampling scenario."""

    name: str
    kind: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or self.kind not in _KINDS:
            raise ValueError("variation needs a name and supported kind")
        expected = 3 if self.kind == "downsample" else 1 if self.kind == "t0" else 2
        if len(self.values) != expected or any(isinstance(v, bool) for v in self.values):
            raise ValueError(f"{self.kind} requires {expected} finite values")
        values = tuple(float(v) for v in self.values)
        if not all(isfinite(v) for v in values):
            raise ValueError(f"{self.kind} requires {expected} finite values")
        if self.kind not in {"t0", "downsample"} and values[1] <= values[0]:
            raise ValueError("window end must exceed start")
        if self.kind == "downsample" and (int(values[0]) != values[0] or values[0] < 3):
            raise ValueError("downsample target must be an integer >= 3")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RobustnessVariation:
        return cls(str(data["name"]), str(data["kind"]), tuple(data["values"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RobustnessRequest:
    source: EcrDataset
    parent_dataset: EcrDataset
    parent_recipe: PreprocessConfig
    parent_config: AnalysisConfig
    parent_fit: FitResult
    parent_fit_id: str
    parent_revision: WorkflowRevision
    parent_fit_generation: int
    variations: tuple[RobustnessVariation, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class RobustnessScenario:
    variation: RobustnessVariation
    revision_id: str
    status: str
    reason: str
    recipe: PreprocessConfig | None = None
    config: AnalysisConfig | None = None
    fit: FitResult | None = None
    fit_id: str | None = None
    selected_model: str | None = None
    dataset_hash: str | None = None
    original_indices: tuple[int, ...] = ()
    affected_indices: tuple[int, ...] = ()
    common_original_indices: tuple[int, ...] = ()
    time_domain: tuple[float, float] | None = None
    changed_parameters: Mapping[str, object] | None = None
    comparability: str = "unavailable"
    comparable_reason: str = "Fit was unavailable."
    common_sse: float | None = None
    parent_common_sse: float | None = None
    residual_summary: Mapping[str, float | None] | None = None
    early_residual_summary: Mapping[str, float | None] | None = None
    residual_change: Mapping[str, float | None] | None = None
    early_residual_change: Mapping[str, float | None] | None = None
    identifiability: str = "unavailable"
    identifiability_change: str = "unavailable"
    quality_warnings: tuple[str, ...] = ()
    warning_stability: Mapping[str, tuple[str, ...]] | None = None
    parameter_changes: Mapping[str, Mapping[str, float | None]] | None = None
    review_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "variation": self.variation.to_dict(), "revision_id": self.revision_id,
            "status": self.status, "reason": self.reason,
            "recipe": None if self.recipe is None else self.recipe.to_dict(),
            "config": None if self.config is None else self.config.to_dict(),
            "fit": None if self.fit is None else self.fit.to_dict(), "fit_id": self.fit_id,
            "selected_model": self.selected_model, "dataset_hash": self.dataset_hash,
            "original_indices": list(self.original_indices), "affected_indices": list(self.affected_indices),
            "common_original_indices": list(self.common_original_indices),
            "time_domain": None if self.time_domain is None else list(self.time_domain),
            "changed_parameters": self.changed_parameters,
            "comparability": self.comparability, "comparable_reason": self.comparable_reason,
            "common_sse": self.common_sse, "parent_common_sse": self.parent_common_sse,
            "residual_summary": self.residual_summary, "early_residual_summary": self.early_residual_summary,
            "residual_change": self.residual_change, "early_residual_change": self.early_residual_change,
            "identifiability": self.identifiability, "identifiability_change": self.identifiability_change,
            "quality_warnings": list(self.quality_warnings), "warning_stability": self.warning_stability,
            "parameter_changes": self.parameter_changes, "review_flags": list(self.review_flags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RobustnessScenario:
        return cls(
            variation=RobustnessVariation.from_dict(data["variation"]),  # type: ignore[arg-type]
            revision_id=str(data["revision_id"]), status=str(data["status"]), reason=str(data["reason"]),
            recipe=None if data.get("recipe") is None else PreprocessConfig.from_dict(data["recipe"]),  # type: ignore[arg-type]
            config=None if data.get("config") is None else AnalysisConfig.from_dict(data["config"]),  # type: ignore[arg-type]
            fit=None if data.get("fit") is None else FitResult.from_dict(data["fit"]),  # type: ignore[arg-type]
            fit_id=data.get("fit_id"), selected_model=data.get("selected_model"),  # type: ignore[arg-type]
            dataset_hash=data.get("dataset_hash"),  # type: ignore[arg-type]
            original_indices=tuple(data.get("original_indices", ())),  # type: ignore[arg-type]
            affected_indices=tuple(data.get("affected_indices", ())),  # type: ignore[arg-type]
            common_original_indices=tuple(data.get("common_original_indices", ())),  # type: ignore[arg-type]
            time_domain=None if data.get("time_domain") is None else tuple(data["time_domain"]),  # type: ignore[arg-type]
            changed_parameters=data.get("changed_parameters"),  # type: ignore[arg-type]
            comparability=str(data.get("comparability", "unavailable")),
            comparable_reason=str(data.get("comparable_reason", "")),
            common_sse=data.get("common_sse"), parent_common_sse=data.get("parent_common_sse"),  # type: ignore[arg-type]
            residual_summary=data.get("residual_summary"),  # type: ignore[arg-type]
            early_residual_summary=data.get("early_residual_summary"),  # type: ignore[arg-type]
            residual_change=data.get("residual_change"),  # type: ignore[arg-type]
            early_residual_change=data.get("early_residual_change"),  # type: ignore[arg-type]
            identifiability=str(data.get("identifiability", "unavailable")),
            identifiability_change=str(data.get("identifiability_change", "unavailable")),
            quality_warnings=tuple(data.get("quality_warnings", ())),  # type: ignore[arg-type]
            warning_stability=data.get("warning_stability"),  # type: ignore[arg-type]
            parameter_changes=data.get("parameter_changes"),  # type: ignore[arg-type]
            review_flags=tuple(data.get("review_flags", ())),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RobustnessAssessment:
    parent_fit_id: str
    parent_revision: WorkflowRevision
    parent_fit_generation: int
    parent_dataset_hash: str
    parent_recipe: PreprocessConfig
    parent_config: AnalysisConfig
    parent_fit: FitResult
    parent_eligible_original_indices: tuple[int, ...]
    parent_residual_summary: Mapping[str, float | None]
    parent_early_residual_summary: Mapping[str, float | None]
    parent_identifiability: str
    parent_quality_warnings: tuple[str, ...]
    seed: int
    scenarios: tuple[RobustnessScenario, ...]
    review_flags: tuple[str, ...]
    schema_version: int = 1  # Standalone artifact version; project schema is unchanged.

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "kind": "p65b_preprocessing_robustness",
            "parent_fit_id": self.parent_fit_id, "parent_revision": vars_revision(self.parent_revision),
            "parent_fit_generation": self.parent_fit_generation,
            "parent_dataset_hash": self.parent_dataset_hash,
            "parent_recipe": self.parent_recipe.to_dict(), "parent_config": self.parent_config.to_dict(),
            "parent_fit": self.parent_fit.to_dict(), "seed": self.seed,
            "parent_eligible_original_indices": list(self.parent_eligible_original_indices),
            "parent_residual_summary": self.parent_residual_summary,
            "parent_early_residual_summary": self.parent_early_residual_summary,
            "parent_identifiability": self.parent_identifiability,
            "parent_quality_warnings": list(self.parent_quality_warnings),
            "scenarios": [item.to_dict() for item in self.scenarios], "review_flags": list(self.review_flags),
            "interpretation": "Processing-choice sensitivity; not a statistical confidence interval.",
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RobustnessAssessment:
        if data.get("schema_version") != 1 or data.get("kind") != "p65b_preprocessing_robustness":
            raise ValueError("unsupported robustness artifact")
        revision = data["parent_revision"]
        if not isinstance(revision, Mapping):
            raise ValueError("parent_revision must be a mapping")
        fit = FitResult.from_dict(data["parent_fit"])  # type: ignore[arg-type]
        if fit_result_id(fit) != data["parent_fit_id"]:
            raise ValueError("parent FitResult hash mismatch")
        scenarios = tuple(RobustnessScenario.from_dict(v) for v in data["scenarios"])  # type: ignore[arg-type]
        for item in scenarios:
            if item.fit is not None and fit_result_id(item.fit) != item.fit_id:
                raise ValueError("scenario FitResult hash mismatch")
            expected = _scenario_revision_id(str(data["parent_fit_id"]),
                WorkflowRevision(**revision), int(data["parent_fit_generation"]),
                item.variation, int(data["seed"]))  # type: ignore[arg-type]
            if item.revision_id != expected:
                raise ValueError("scenario revision ID mismatch")
        return cls(
            parent_fit_id=str(data["parent_fit_id"]),
            parent_revision=WorkflowRevision(**revision),  # type: ignore[arg-type]
            parent_fit_generation=int(data["parent_fit_generation"]),
            parent_dataset_hash=str(data["parent_dataset_hash"]),
            parent_recipe=PreprocessConfig.from_dict(data["parent_recipe"]),  # type: ignore[arg-type]
            parent_config=AnalysisConfig.from_dict(data["parent_config"]),  # type: ignore[arg-type]
            parent_fit=fit,
            parent_eligible_original_indices=tuple(data.get("parent_eligible_original_indices", ())),  # type: ignore[arg-type]
            parent_residual_summary=data.get("parent_residual_summary", {}),  # type: ignore[arg-type]
            parent_early_residual_summary=data.get("parent_early_residual_summary", {}),  # type: ignore[arg-type]
            parent_identifiability=str(data.get("parent_identifiability", "unavailable")),
            parent_quality_warnings=tuple(data.get("parent_quality_warnings", ())),  # type: ignore[arg-type]
            seed=int(data["seed"]), scenarios=scenarios,
            review_flags=tuple(data.get("review_flags", ())),  # type: ignore[arg-type]
        )

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> RobustnessAssessment:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def vars_revision(revision: WorkflowRevision) -> dict[str, int]:
    return {"data_revision": revision.data_revision, "config_revision": revision.config_revision,
            "physical_input_revision": revision.physical_input_revision}


def _scenario_revision_id(parent_id: str, revision: WorkflowRevision, generation: int,
                          variation: RobustnessVariation, seed: int) -> str:
    return hashlib.sha256(json.dumps({"parent": parent_id, "revision": vars_revision(revision),
        "generation": generation, "variation": variation.to_dict(), "seed": seed},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_robustness(session: AnalysisSession, variations: Sequence[RobustnessVariation]) -> RobustnessRequest:
    """Capture a current parent without changing workflow state."""
    if (session.preprocessing is None or session.dataset is None or session.current_analysis_config is None
            or session.fit_result is None or session.fit_stale or session.fit_revision != session.current_revision):
        raise RuntimeError("Robustness assessment requires a current non-stale Fit")
    choices = tuple(variations)
    if not 1 <= len(choices) <= _MAX_SCENARIOS or len({v.name for v in choices}) != len(choices):
        raise ValueError("choose 1–8 uniquely named scenarios")
    if any(not isinstance(v, RobustnessVariation) for v in choices):
        raise TypeError("variations must be RobustnessVariation objects")
    is_surface_k = session.current_analysis_config.model_id is ModelID.M7_SINGLE_K
    options_name = "surface_k" if is_surface_k else "single_ecr" if session.current_analysis_config.model_id in {ModelID.M0_SINGLE_D, ModelID.M1_SINGLE_DK} else "dual_ho"
    options = session.current_analysis_config.model_specific.get(options_name, {})
    seed = int(options.get("random_seed", 20260915)) if isinstance(options, Mapping) else 20260915
    if is_surface_k and session.fit_result.random_seed is not None:
        seed = session.fit_result.random_seed
    return RobustnessRequest(session.preprocessing.source, session.dataset, session.preprocessing.config,
                             session.current_analysis_config, session.fit_result, fit_result_id(session.fit_result),
                             session.current_revision, session.fit_generation, choices, seed)


def run_robustness(
    request: RobustnessRequest, *, analysis_service: AnalysisService | None = None,
    diagnostics_service: DiagnosticsService | None = None, cancel_token: CancellationToken | None = None,
    progress_callback=None,
) -> RobustnessAssessment:
    """Run isolated production fits; return only after every scenario settles."""
    if not isinstance(request, RobustnessRequest) or not 1 <= len(request.variations) <= _MAX_SCENARIOS:
        raise ValueError("run_robustness requires a captured request with 1–8 scenarios")
    token = cancel_token or CancellationToken()
    token.raise_if_cancelled()
    analysis = analysis_service or AnalysisService()
    diagnostics = diagnostics_service or DiagnosticsService(analysis_service=analysis)
    parent_quality = evaluate_fit_quality(request.parent_dataset, request.parent_config)
    try:
        parent_ident = diagnostics.describe(request.parent_dataset, request.parent_config,
                                             request.parent_fit).reliability_advisor.identifiability_status
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        parent_ident = f"unavailable: {exc}"
    token.raise_if_cancelled()
    parent_indices = tuple(int(i) for i in np.flatnonzero(request.parent_dataset.mask))
    parent_residuals = _residual_stats(request.parent_fit.residuals)
    parent_early = _early_stats(request.parent_dataset.processed_time, request.parent_fit.residuals)
    scenarios: list[RobustnessScenario] = []
    for number, variation in enumerate(request.variations, 1):
        token.raise_if_cancelled()
        if progress_callback is not None:
            progress_callback((number - 1) / len(request.variations), f"Robustness {number}/{len(request.variations)}: {variation.name}")
        revision_id = _scenario_revision_id(request.parent_fit_id, request.parent_revision,
                                            request.parent_fit_generation, variation, request.seed)
        try:
            recipe, changed, affected, domain = _variant_recipe(request, variation)
            token.raise_if_cancelled()
            dataset = recipe.replay(request.source)
            indices = tuple(int(i) for i in np.flatnonzero(dataset.mask))
            config = AnalysisConfig.from_dict(request.parent_config.to_dict())
            if config.model_id is ModelID.M7_SINGLE_K and request.parent_fit.random_seed is not None:
                from dataclasses import replace
                config = replace(config, model_specific={**config.model_specific,
                    "surface_k": {**config.model_specific.get("surface_k", {}), "random_seed": request.seed}})
            quality = evaluate_fit_quality(dataset, config)
            if not quality.ready:
                raise ValueError("Fit Quality Gate: " + "; ".join(quality.blocking_reasons))
            token.raise_if_cancelled()
            fit = analysis.fit(dataset, config, cancel_token=token,
                               progress_callback=lambda _fraction, _message: token.raise_if_cancelled())
            token.raise_if_cancelled()
            selected = _selected_model(config, fit)
            parent_selected = _selected_model(request.parent_config, request.parent_fit)
            comparable, why = _comparison_class(request, dataset, variation, selected == parent_selected)
            common_sse = parent_common = None
            if comparable in {"same_processed_rows", "translated_time_rows"}:
                # Subsampling is always re-evaluated on the full parent fit-eligible set.
                reference = request.parent_dataset if variation.kind == "downsample" else dataset
                try:
                    common_sse = _production_sse(reference, config, fit, selected, token)
                    parent_common = _production_sse(request.parent_dataset, request.parent_config,
                                                    request.parent_fit, parent_selected, token)
                except (ValueError, TypeError, ArithmeticError, RuntimeError) as exc:
                    token.raise_if_cancelled()
                    comparable = "unavailable_common_objective"
                    why = f"Production reference objective could not evaluate both parameter sets: {exc}"
                    common_sse = parent_common = None
            token.raise_if_cancelled()
            ident = "unavailable"
            try:
                ident = diagnostics.describe(dataset, config, fit).reliability_advisor.identifiability_status
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                ident = f"unavailable: {exc}"
            token.raise_if_cancelled()
            scenarios.append(RobustnessScenario(
                variation=variation, revision_id=revision_id, status="completed", reason="",
                recipe=recipe, config=config, fit=fit, fit_id=fit_result_id(fit), selected_model=selected,
                dataset_hash=dataset_sha256(dataset), original_indices=indices, affected_indices=affected,
                common_original_indices=tuple(sorted(set(indices).intersection(parent_indices))),
                time_domain=domain, changed_parameters=changed, comparability=comparable,
                comparable_reason=why, common_sse=common_sse, parent_common_sse=parent_common,
                residual_summary=_residual_stats(fit.residuals),
                early_residual_summary=_early_stats(dataset.processed_time, fit.residuals),
                residual_change=_summary_change(parent_residuals, _residual_stats(fit.residuals)),
                early_residual_change=_summary_change(parent_early,
                    _early_stats(dataset.processed_time, fit.residuals)),
                identifiability=ident, identifiability_change=f"{parent_ident} → {ident}",
                quality_warnings=quality.warnings,
                warning_stability={"retained": tuple(sorted(set(parent_quality.warnings) & set(quality.warnings))),
                                   "new": tuple(sorted(set(quality.warnings) - set(parent_quality.warnings))),
                                   "resolved": tuple(sorted(set(parent_quality.warnings) - set(quality.warnings)))},
                parameter_changes=_parameter_changes(request.parent_fit, fit),
                review_flags=_review_flags(recipe, dataset),
            ))
        except Exception as exc:
            token.raise_if_cancelled()
            scenarios.append(RobustnessScenario(variation, revision_id, "failed", f"{type(exc).__name__}: {exc}"))
    token.raise_if_cancelled()
    if progress_callback is not None:
        progress_callback(1.0, "Robustness assessment complete")
    return RobustnessAssessment(
        parent_fit_id=request.parent_fit_id, parent_revision=request.parent_revision,
        parent_fit_generation=request.parent_fit_generation,
        parent_dataset_hash=dataset_sha256(request.parent_dataset),
        parent_recipe=request.parent_recipe, parent_config=request.parent_config,
        parent_fit=request.parent_fit, parent_eligible_original_indices=parent_indices,
        parent_residual_summary=parent_residuals,
        parent_early_residual_summary=parent_early,
        parent_identifiability=parent_ident, parent_quality_warnings=parent_quality.warnings,
        seed=request.seed, scenarios=tuple(scenarios),
        review_flags=_review_flags(request.parent_recipe, request.parent_dataset),
    )


def _variant_recipe(request: RobustnessRequest, variation: RobustnessVariation) -> tuple[PreprocessConfig, dict[str, object], tuple[int, ...], tuple[float, float]]:
    steps = list(request.parent_recipe.steps)
    if variation.kind == "downsample":
        target, early, power = variation.values
        step = preview_downsampling(request.parent_dataset, target_points=int(target), early_until=early, late_power=power).step
        scenario_steps = [*steps, step]
        # A fixed normalization reference survives thinning. Re-recording a
        # smaller plateau after normalization would replace calibration evidence.
        # Unnormalized views still need live, complete selection descriptors.
        fixed_reference = request.parent_dataset.normalization_reference() is not None
        windows = request.parent_dataset.metadata.get("stability_windows")
        revalidated: list[str] = []
        if isinstance(windows, Mapping) and not fixed_reference:
            for kind in ("initial", "final"):
                descriptor = windows.get(kind)
                if isinstance(descriptor, Mapping):
                    operation = f"choose_{kind}_stability_window"
                    original = next((s for s in reversed(steps) if s.operation == operation), None)
                    if original is None:
                        raise ValueError(f"{kind} stability window has no recipe step to replay")
                    start, end = request.parent_dataset.stability_window_bounds(kind, descriptor)
                    scenario_steps.append(ProcessingStep.create(operation,
                        {"start_time": start, "end_time": end}, timestamp=original.timestamp))
                    revalidated.append(kind)
        new = PreprocessConfig(tuple(scenario_steps))
        affected = tuple(int(i) for i in step.parameters["input_indices"] if i not in step.parameters["selected_indices"])
        return new, {"new": variation.to_dict(), "old": None,
                     "metadata_only_window_revalidation": revalidated,
                     "fixed_normalization_reference_retained": fixed_reference,
                     "reversible": "discard scenario; replay parent recipe"}, affected, (float(request.parent_dataset.processed_time[0]), float(request.parent_dataset.processed_time[-1]))
    operation = {"initial_window": "choose_initial_stability_window", "final_window": "choose_final_stability_window",
                 "t0": "set_manual_t0", "drift_reference": "drift_correction"}[variation.kind]
    matches = [i for i, step in enumerate(steps) if step.operation == operation]
    if len(matches) != 1:
        raise ValueError(f"{variation.kind} requires exactly one existing {operation} step")
    position = matches[0]
    old = steps[position]
    prefix = PreprocessConfig(tuple(steps[:position])).replay(request.source)
    if variation.kind == "t0":
        # Shift later time-domain descriptors by exactly the same translation,
        # preserving their original raw-row selections. This is a scenario
        # recipe only; the accepted parent recipe remains untouched.
        if any(s.operation == "convert_time_unit" for s in steps[position + 1:]):
            raise ValueError("t0 perturbation with a later time-unit conversion has no defined mapping")
        shift = variation.values[0] - float(old.parameters["t0"])
        for index in range(position + 1, len(steps)):
            later = steps[index]
            p = later.to_dict()["parameters"]
            if later.operation in {"crop", "choose_initial_stability_window", "choose_final_stability_window"}:
                p["start_time"] = float(p["start_time"]) - shift
                p["end_time"] = float(p["end_time"]) - shift
            elif later.operation == "drift_correction":
                p["reference_window"] = [float(v) - shift for v in p["reference_window"]]
                p["scope"] = [float(v) - shift for v in p["scope"]]
                p["anchor_time"] = float(p["anchor_time"]) - shift
            elif later.operation == "adaptive_downsample":
                p["early_until"] = float(p["early_until"]) - shift
            else:
                continue
            steps[index] = ProcessingStep.create(later.operation, p, timestamp=later.timestamp)
        step = ProcessingStep.create(operation, {"t0": variation.values[0]}, timestamp=old.timestamp)
        affected = tuple(int(i) for i in np.flatnonzero(prefix.mask))
        domain = (float(prefix.processed_time[0]), float(prefix.processed_time[-1]))
    elif variation.kind == "drift_reference":
        old_params = old.parameters
        preview = preview_drift(prefix, reference_window=variation.values,
                                scope=old_params["scope"], method=str(old_params["method"]),
                                anchor_time=float(old_params["anchor_time"]))
        step = ProcessingStep.create(operation, preview.step.parameters, timestamp=old.timestamp)
        affected = tuple(int(i) for i in step.parameters["affected_raw_indices"])
        domain = tuple(float(v) for v in step.parameters["scope"])
    else:
        step = ProcessingStep.create(operation, {"start_time": variation.values[0],
                                                       "end_time": variation.values[1]}, timestamp=old.timestamp)
        selected = (prefix.processed_time >= variation.values[0]) & (prefix.processed_time <= variation.values[1])
        affected = tuple(int(i) for i in np.flatnonzero(prefix.mask)[selected])
        domain = (variation.values[0], variation.values[1])
    steps[position] = step
    return PreprocessConfig(tuple(steps)), {"old": old.to_dict()["parameters"], "new": step.to_dict()["parameters"],
            "step_index": position, "downstream_time_translation": None if variation.kind != "t0" else -shift,
            "reversible": "discard scenario; replay parent recipe"}, affected, domain


def _selected_model(config: AnalysisConfig, fit: FitResult) -> str:
    dispatch = fit.diagnostics.get("analysis_dispatch")
    return str(dispatch.get("selected_model_id", config.model_id.value)) if isinstance(dispatch, Mapping) else config.model_id.value


def _comparison_class(request: RobustnessRequest, candidate: EcrDataset, variation: RobustnessVariation,
                      same_model: bool) -> tuple[str, str]:
    if not same_model:
        return "incomparable_model", "Production model selection differs between parent and scenario."
    parent = request.parent_dataset
    if (candidate.metadata.get("signal_type") != parent.metadata.get("signal_type") or
            candidate.metadata.get("signal_unit") != parent.metadata.get("signal_unit")):
        return "incomparable_observation_space", "Signal type or unit changed."
    raw_parent, raw_candidate = np.flatnonzero(parent.mask), np.flatnonzero(candidate.mask)
    if variation.kind == "downsample":
        retained = np.isin(raw_parent, raw_candidate)
        if (not set(raw_candidate).issubset(set(raw_parent)) or
                not np.array_equal(candidate.processed_signal, parent.processed_signal[retained]) or
                not np.array_equal(candidate.processed_time, parent.processed_time[retained])):
            return "incomparable_observation_space", "Downsample changed more than row inclusion."
        return "same_processed_rows", "Both parameter sets re-evaluated on all parent fit-eligible rows with the production objective."
    if not np.array_equal(raw_parent, raw_candidate):
        return "incomparable_observation_space", "Original row set changed."
    if not np.array_equal(parent.processed_signal, candidate.processed_signal):
        return "incomparable_observation_space", "Processed signal values differ; SSE scales cannot be directly ranked."
    if variation.kind == "t0":
        delta = candidate.processed_time - parent.processed_time
        tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(parent.processed_time))))
        if not np.allclose(delta, delta[0], rtol=0, atol=tolerance):
            return "incomparable_time_mapping", "t0 did not produce a constant row-wise time translation."
        return "translated_time_rows", f"Same observed rows and signal; candidate time = parent time + {delta[0]:.17g} s."
    if not np.array_equal(parent.processed_time, candidate.processed_time):
        return "incomparable_time_mapping", "Processed time mapping differs."
    return "same_processed_rows", "Identical processed observations and time coordinates."


def _production_sse(dataset: EcrDataset, config: AnalysisConfig, fit: FitResult,
                    selected_model: str, token: CancellationToken) -> float:
    token.raise_if_cancelled()
    if selected_model == ModelID.M7_SINGLE_K.value:
        model = ModelFactory().create(config)
        parameters = {**fit.diagnostics["surface_k"]["fixed_parameters"], **fit.params}
        residuals = dataset.processed_signal - model.predict(dataset.processed_time, parameters)
        value = float(residuals @ residuals)
    elif selected_model in {ModelID.M0_SINGLE_D.value, ModelID.M1_SINGLE_DK.value}:
        objective = prepare_single_objective(dataset, config, cancel_check=token.raise_if_cancelled)
        coordinates = []
        for name in objective.context.parameter_names:
            physical = {"log10_D": "D_cm2_s", "log10_k": "k_cm_s"}.get(name, name)
            value = float(fit.params[physical])
            coordinates.append(log10(value) if name.startswith("log10_") else value)
        value = objective.evaluate(coordinates).sse
    elif selected_model in {ModelID.M2_DUAL_HO_D.value, ModelID.M3_DUAL_HO_DK.value}:
        fit_config, _ = _dual_fit_adapter(config)
        model = ModelFactory().create(config)
        if not isinstance(model, DualHOModel):
            raise TypeError("production dual model unavailable")
        is_m3 = selected_model == ModelID.M3_DUAL_HO_DK.value
        parameterization = _Parameterization(fit_config, dataset.processed_signal, is_m3=is_m3)
        context = _context_for_level(model, dataset.processed_time, dataset.processed_signal,
            parameterization=parameterization, fit_config=fit_config, is_m3=is_m3,
            geometry=config.model.geometry, signal_type=str(dataset.metadata.get("signal_type")),
            signal_unit=str(dataset.metadata.get("signal_unit")))
        value = DualHOObjectiveEvaluator(context).evaluate_raw_sse(fit.params)
    else:
        raise ValueError(f"unsupported selected model: {selected_model}")
    token.raise_if_cancelled()
    return value


def _residual_stats(residuals: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(residuals, dtype=float)
    lag = None
    if values.size >= 3 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
        lag = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        if not isfinite(lag):
            lag = None
    return {"count": int(values.size), "mean": float(np.mean(values)),
            "rms": sqrt(float(np.mean(values * values))), "lag1": lag}


def _early_stats(times: np.ndarray, residuals: np.ndarray) -> dict[str, float | None]:
    # First third of the observed duration, including the first observed row.
    threshold = float(times[0] + (times[-1] - times[0]) / 3)
    selected = np.asarray(times) <= threshold
    result = _residual_stats(np.asarray(residuals)[selected])
    result["end_time_s"] = threshold
    return result


def _parameter_changes(parent: FitResult, child: FitResult) -> dict[str, dict[str, float | None]]:
    result = {}
    for key in _KINETIC:
        if key in parent.params and key in child.params:
            before, after = float(parent.params[key]), float(child.params[key])
            result[key] = {"parent": before, "scenario": after, "delta": after - before,
                           "relative_difference": None if before == 0 else (after - before) / abs(before)}
    return result


def _summary_change(parent: Mapping[str, float | None], child: Mapping[str, float | None]) -> dict[str, float | None]:
    return {f"delta_{key}": None if parent.get(key) is None or child.get(key) is None
            else float(child[key]) - float(parent[key]) for key in ("mean", "rms", "lag1")}


def _review_flags(recipe: PreprocessConfig, dataset: EcrDataset) -> tuple[str, ...]:
    flags: list[str] = []
    for step in recipe.steps:
        p = step.parameters
        if step.operation == "drift_correction":
            if p.get("method") == "robust_linear":
                flags.append(f"Theil-Sen uses {p.get('reference_points')} reference points; current cap is 3000.")
            scope = p.get("scope")
            if isinstance(scope, (tuple, list)) and (float(scope[0]) > float(dataset.processed_time[0]) or
                                                    float(scope[1]) < float(dataset.processed_time[-1])):
                slope, anchor = float(p.get("slope", 0)), float(p.get("anchor_time", 0))
                edges = [abs(slope * (float(edge) - anchor)) for edge in scope]
                flags.append("Local drift scope may create an edge jump; estimated correction at scope edges "
                             f"is {edges[0]:.6g}/{edges[1]:.6g} signal units. Inspect both edges.")
        elif step.operation == "outlier_mask":
            marked = tuple(int(i) for i in p.get("raw_indices", ()))
            transition = int(np.argmax(np.abs(np.diff(dataset.raw_signal) / np.diff(dataset.raw_time))))
            nearby = [i for i in marked if abs(i - transition) <= 2]
            if nearby:
                flags.append(f"Marked original indices {nearby} touch the strongest raw transition; verify a real step was not masked.")
            else:
                flags.append("Review marked original indices against the physical step: an abrupt real transition may look like an outlier.")
            flags.append("The local-extremum detector can miss consecutive anomalies; inspect neighboring unmasked points.")
    return tuple(dict.fromkeys(flags))
