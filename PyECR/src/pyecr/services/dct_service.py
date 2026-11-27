"""Application service for diagnostic characteristic-time distributions.

This layer validates workflow inputs, calls the existing DCT core, and prepares
project/GUI data.  It never assigns a recovered peak to a chemical species or
changes the active physical model configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log10
from types import MappingProxyType
from typing import Mapping

from pyecr.config import SampleGeometry
from pyecr.dct.core import DCTResult, fit_dct
from pyecr.dct.config import DctAnalysisConfig
from pyecr.diagnostics.dct_assistant import (
    DCTModelSuggestion,
    suggest_models_from_dct,
)
from pyecr.io.dataset import EcrDataset
from pyecr.services.analysis import CancellationToken, ProgressReporter

__all__ = [
    "DctAnalysisConfig",
    "DctPresentation",
    "DctService",
    "DctSuggestionPresentation",
]


@dataclass(frozen=True, slots=True)
class DctSuggestionPresentation:
    """Serializable diagnostic model trials; never an automatic selection."""

    recommended_model_ids: tuple[str, ...]
    recommendation: str
    selected_peak_count: int
    dominant_peak_count: int
    stable_peak_count: int
    stable_peak_positions_s: tuple[float, ...]
    lambda_stability: Mapping[str, object]
    artifact_risk: bool
    artifact_reasons: tuple[str, ...]
    classical_bi_assessment: Mapping[str, object] | None
    geometry_summary: Mapping[str, object]
    chemical_species_assignment: bool
    warnings: tuple[str, ...]
    reasoning: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.chemical_species_assignment:
            raise ValueError("DCT suggestions must not assign chemical species")
        object.__setattr__(
            self, "recommended_model_ids", tuple(self.recommended_model_ids)
        )
        object.__setattr__(
            self, "stable_peak_positions_s", tuple(self.stable_peak_positions_s)
        )
        object.__setattr__(self, "artifact_reasons", tuple(self.artifact_reasons))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reasoning", tuple(self.reasoning))
        object.__setattr__(
            self,
            "lambda_stability",
            MappingProxyType(dict(self.lambda_stability)),
        )
        object.__setattr__(
            self,
            "geometry_summary",
            MappingProxyType(dict(self.geometry_summary)),
        )
        if self.classical_bi_assessment is not None:
            legacy = dict(self.classical_bi_assessment)
            legacy.setdefault('method_version', 'legacy-unversioned')
            legacy['classification'] = 'legacy_heuristic'
            legacy['quantitative_bi'] = False
            object.__setattr__(
                self,
                "classical_bi_assessment",
                MappingProxyType(legacy),
            )

    @classmethod
    def from_core(
        cls, suggestion: DCTModelSuggestion
    ) -> DctSuggestionPresentation:
        classical = suggestion.classical_bi_assessment
        return cls(
            recommended_model_ids=tuple(
                item.value for item in suggestion.recommended_model_ids
            ),
            recommendation=suggestion.recommendation,
            selected_peak_count=suggestion.selected_peak_count,
            dominant_peak_count=suggestion.dominant_peak_count,
            stable_peak_count=suggestion.stable_peak_count,
            stable_peak_positions_s=suggestion.stable_peak_positions_s,
            lambda_stability=_json_mapping(asdict(suggestion.lambda_stability)),
            artifact_risk=suggestion.artifact_risk,
            artifact_reasons=suggestion.artifact_reasons,
            classical_bi_assessment=(
                None if classical is None else {**_json_mapping(asdict(classical)), 'method_version':'legacy-peak-ratio/1'}
            ),
            geometry_summary=_json_mapping(suggestion.geometry_summary),
            chemical_species_assignment=False,
            warnings=suggestion.warnings,
            reasoning=suggestion.reasoning,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recommended_model_ids": list(self.recommended_model_ids),
            "recommendation": self.recommendation,
            "selected_peak_count": self.selected_peak_count,
            "dominant_peak_count": self.dominant_peak_count,
            "stable_peak_count": self.stable_peak_count,
            "stable_peak_positions_s": list(self.stable_peak_positions_s),
            "lambda_stability": _json_mapping(self.lambda_stability),
            "artifact_risk": self.artifact_risk,
            "artifact_reasons": list(self.artifact_reasons),
            "classical_bi_assessment": (
                None
                if self.classical_bi_assessment is None
                else _json_mapping(self.classical_bi_assessment)
            ),
            "geometry_summary": _json_mapping(self.geometry_summary),
            "chemical_species_assignment": False,
            "warnings": list(self.warnings),
            "reasoning": list(self.reasoning),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> DctSuggestionPresentation:
        return cls(
            recommended_model_ids=_string_tuple(value, "recommended_model_ids"),
            recommendation=str(value["recommendation"]),
            selected_peak_count=int(value["selected_peak_count"]),
            dominant_peak_count=int(value["dominant_peak_count"]),
            stable_peak_count=int(value["stable_peak_count"]),
            stable_peak_positions_s=_float_tuple(
                value, "stable_peak_positions_s"
            ),
            lambda_stability=_required_mapping(value, "lambda_stability"),
            artifact_risk=bool(value["artifact_risk"]),
            artifact_reasons=_string_tuple(value, "artifact_reasons"),
            classical_bi_assessment=(
                None
                if value.get("classical_bi_assessment") is None
                else _required_mapping(value, "classical_bi_assessment")
            ),
            geometry_summary=_required_mapping(value, "geometry_summary"),
            chemical_species_assignment=bool(
                value.get("chemical_species_assignment", False)
            ),
            warnings=_string_tuple(value, "warnings"),
            reasoning=_string_tuple(value, "reasoning"),
        )


@dataclass(frozen=True, slots=True)
class DctPresentation:
    """Complete GUI/project view of one DCT diagnostic run."""

    config: DctAnalysisConfig
    time_s: tuple[float, ...]
    signal: tuple[float, ...]
    result: DCTResult
    suggestion: DctSuggestionPresentation
    schema_version: int = 1

    @property
    def log10_tau(self) -> tuple[float, ...]:
        return tuple(log10(float(value)) for value in self.result.tau_s)

    @property
    def chi(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.result.chi)

    @property
    def reconstruction(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.result.y_fit)

    @property
    def residuals(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.result.residuals)

    @property
    def peak_positions_s(self) -> tuple[float, ...]:
        return self.result.peak_positions_s

    @property
    def peak_areas(self) -> tuple[float, ...]:
        return self.result.peak_areas

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "time_s": list(self.time_s),
            "signal": list(self.signal),
            "result": self.result.to_dict(),
            "suggestion": self.suggestion.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DctPresentation:
        if "contract_kind" in value or "mode" in value:
            raise ValueError("Versioned/new-mode DCT requires the explicit contract reader; legacy fallback refused")
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("unsupported DctPresentation schema_version")
        return cls(
            config=DctAnalysisConfig.from_dict(_required_mapping(value, "config")),
            time_s=_float_tuple(value, "time_s"),
            signal=_float_tuple(value, "signal"),
            result=DCTResult.from_dict(_required_mapping(value, "result")),
            suggestion=DctSuggestionPresentation.from_dict(
                _required_mapping(value, "suggestion")
            ),
        )


class DctService:
    def __init__(self, *, isolate_numeric=False):
        self.isolate_numeric = bool(isolate_numeric)

    """Run DCT and produce diagnostic-only model trial suggestions."""

    def analyze(
        self,
        dataset: EcrDataset,
        geometry: SampleGeometry | None,
        config: DctAnalysisConfig,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
    ) -> DctPresentation:
        if not isinstance(dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        if geometry is not None and not isinstance(geometry, SampleGeometry):
            raise TypeError("geometry must be a SampleGeometry")
        if not isinstance(config, DctAnalysisConfig):
            raise TypeError("config must be a DctAnalysisConfig")
        config.validate()
        if geometry is not None: geometry.validate()
        if dataset.metadata.get("normalization_mode") == "signed_terminal":
            raise ValueError(
                "Signed-DCT normalization requires the explicit signed DCT mode; "
                "the legacy nonnegative solver cannot interpret signed_terminal data"
            )
        signal_type = str(dataset.metadata.get("signal_type", "unknown")).lower()
        if signal_type != "normalized":
            raise ValueError(
                "DCT requires a normalized processed signal; preprocess explicitly "
                "before running this diagnostic"
            )
        if str(dataset.metadata.get("time_unit", "s")) != "s":
            raise ValueError("DCT requires processed time in seconds")
        token = cancel_token or CancellationToken()
        progress = progress_callback or (lambda fraction, message: None)
        token.raise_if_cancelled()
        progress(0.05, "Validating normalized DCT input")
        from pyecr.dct.execution import run_numeric
        result = run_numeric(self.isolate_numeric, token.raise_if_cancelled, fit_dct,
            dataset.processed_time,
            dataset.processed_signal,
            tau_min_s=config.tau_min_s,
            tau_max_s=config.tau_max_s,
            n_tau=config.n_tau,
            gcv_lambdas=config.gcv_lambdas or None,
            regularization_lambda=(
                config.regularization_lambda
                if config.lambda_mode == "manual"
                else None
            ),
        )
        token.raise_if_cancelled()
        progress(0.90, "Preparing candidate time-scale suggestions")
        suggestion = suggest_models_from_dct(
            result,
            geometry,
            use_classical_bi=config.use_classical_bi,
        )
        token.raise_if_cancelled()
        progress(1.0, "DCT diagnostic complete")
        return DctPresentation(
            config=config,
            time_s=tuple(float(value) for value in dataset.processed_time),
            signal=tuple(float(value) for value in dataset.processed_signal),
            result=result,
            suggestion=DctSuggestionPresentation.from_core(suggestion),
        )


def _required_mapping(
    value: Mapping[str, object], name: str
) -> Mapping[str, object]:
    item = value.get(name)
    if not isinstance(item, Mapping):
        raise TypeError(f"{name} must be an object")
    return item


def _sequence(value: Mapping[str, object], name: str) -> list[object] | tuple[object, ...]:
    item = value.get(name)
    if not isinstance(item, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return item


def _float_tuple(value: Mapping[str, object], name: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _sequence(value, name))


def _string_tuple(value: Mapping[str, object], name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, name))


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_value(item) for key, item in value.items()}
