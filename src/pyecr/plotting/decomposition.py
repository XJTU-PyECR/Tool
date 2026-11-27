"""Continuous signed contributions of one completed joint fit (no inference)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.io.dataset import EcrDataset
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID

VERSION = "continuous-decomposition/1.0.0"
GRID_RULE = "observations-plus-linear-log1p-v1"
DUAL_IDS = {"M2_DUAL_HO_D", "M3_DUAL_HO_DK", "M8_DUAL_K"}
INTERPRETATION = (
    "Signed effective contributions to the measured signal; not independent carrier "
    "conductivities. Reconstruction does not establish unique physical attribution. "
    "No component confidence intervals are supplied."
)


@dataclass(frozen=True, slots=True)
class DecompositionViewOptions:
    version: str = VERSION
    mode: str = "zero"
    show_early_inset: bool = False
    show_residuals: bool = True
    show_amplitudes: bool = False
    early_fraction: float = .15
    display_points: int = 512

    def __post_init__(self):
        if self.version != VERSION or self.mode not in {"zero", "shifted"}:
            raise ValueError("unsupported decomposition view version/mode")
        for name in ("show_early_inset", "show_residuals", "show_amplitudes"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if isinstance(self.early_fraction, bool) or not isfinite(self.early_fraction) or not 0 < self.early_fraction <= 1:
            raise ValueError("early_fraction must be in (0, 1]")
        if isinstance(self.display_points, bool) or not isinstance(self.display_points, int) or not 32 <= self.display_points <= 4096:
            raise ValueError("display_points must be an integer in [32, 4096]")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping) or "version" not in value:
            raise ValueError("decomposition view requires a versioned object")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class DecompositionGrid:
    time_s: tuple[float, ...]
    total: tuple[float, ...]
    component_1: tuple[float, ...]
    component_2: tuple[float, ...]

    def __post_init__(self):
        for name in ("time_s", "total", "component_1", "component_2"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values or not all(isfinite(v) for v in values):
                raise ValueError(f"{name} must be a nonempty finite vector")
            object.__setattr__(self, name, values)
        if any(len(getattr(self, n)) != len(self.time_s) for n in ("total", "component_1", "component_2")):
            raise ValueError("decomposition arrays must align")
        if any(b <= a for a, b in zip(self.time_s, self.time_s[1:])):
            raise ValueError("decomposition time must be strictly increasing")

    def contributions(self, y0: float, mode: str):
        if mode == "zero":
            return self.component_1, self.component_2
        if mode != "shifted":
            raise ValueError("unknown decomposition mode")
        return tuple(y0 + x for x in self.component_1), tuple(y0 + x for x in self.component_2)


@dataclass(frozen=True, slots=True)
class DecompositionData:
    """Observation grid is authoritative; display grid never defines residuals."""
    model_id: str
    observation: DecompositionGrid
    display: DecompositionGrid
    observed: tuple[float, ...]
    residuals: tuple[float, ...]
    original_raw_indices: tuple[int, ...]
    y0: float
    signed_amplitudes: tuple[float, float]
    labels: tuple[str, str]
    label_semantics: str
    signal_unit: str
    signal_type: str
    dataset_sha256: str
    config_sha256: str
    fit_result_sha256: str
    options: DecompositionViewOptions
    observation_parity_max_abs: float

    def __post_init__(self):
        for name in ("observed", "residuals", "signed_amplitudes"):
            values = tuple(float(v) for v in getattr(self, name))
            if not all(isfinite(v) for v in values):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "original_raw_indices", tuple(self.original_raw_indices))
        n = len(self.observation.time_s)
        if self.model_id not in DUAL_IDS or not all(len(v) == n for v in (self.observed, self.residuals, self.original_raw_indices)):
            raise ValueError("decomposition requires a complete aligned dual FitResult")
        if not isfinite(self.y0) or len(self.labels) != 2 or len(self.signed_amplitudes) != 2:
            raise ValueError("decomposition baseline, amplitudes and labels are required")
        for grid in (self.observation, self.display):
            _parity(np.asarray(grid.total), self.y0 + np.asarray(grid.component_1) + grid.component_2, "zero reconstruction")
            a, b = grid.contributions(self.y0, "shifted")
            _parity(np.asarray(grid.total), np.asarray(a) + b - self.y0, "shifted reconstruction")

    @property
    def early_end_s(self):
        times = self.observation.time_s
        return times[0] + self.options.early_fraction * (times[-1] - times[0])

    @property
    def formula(self):
        return "yfit = y0 + component_1 + component_2" if self.options.mode == "zero" else "yfit = shifted_1 + shifted_2 - y0"

    def to_dict(self):
        return {
            **asdict(self), "version": VERSION, "display_grid_rule": GRID_RULE,
            "interpretation": INTERPRETATION,
            "component_definition": "signed change from the common initial baseline",
            "shifted_definition": "shifted_j = y0 + component_j",
            "scientific_grid": "complete fit-eligible observations; no interpolated residuals",
            "component_ci": {"available": False, "reason": "not_calibrated"},
        }


def _parity(actual, expected, name):
    # Cancellation may involve large individual contributions and a tiny total.
    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))), 1.)
    if not np.allclose(actual, expected, rtol=2e-12, atol=2e-12 * scale):
        raise ValueError(f"decomposition {name} mismatch; current model/result required")


def _observation_context(dataset: EcrDataset, config: AnalysisConfig, fit: FitResult):
    """One production observation extraction shared by display and source capture.

    The selected legacy model may be M2 even when M3 was requested. Its actual
    baseline defaults and kernel options are taken from the original adapter.
    No fit, normalization, segmentation, interpolation or uncertainty is run.
    """
    from pyecr.report.snapshot import dataset_sha256, _sha256_json, _validate_surface_k_binding
    from pyecr.services.analysis import _dual_fit_adapter
    from pyecr.fitting.reproducibility import effective_config
    config=effective_config(config,fit)

    config.validate()
    if config.model_id.value not in DUAL_IDS or fit.exit_status <= 0:
        raise ValueError("decomposition requires a completed dual joint fit")
    t, y = dataset.processed_time, dataset.processed_signal
    if dataset.metadata.get("time_unit", "s") != "s" or len(t) != len(fit.y_fit):
        raise ValueError("decomposition requires aligned fit-eligible data in seconds")
    # Detect stale/incompatible full-data observations before any display evaluation.
    _parity(fit.residuals, y - fit.y_fit, "full observation binding")
    if not np.isclose(fit.SSE, float(fit.residuals @ fit.residuals), rtol=1e-10, atol=1e-14):
        raise ValueError("decomposition full-data raw SSE binding mismatch")
    if config.model_id is ModelID.M8_DUAL_K or "fixed_d" in config.model_specific:
        _validate_surface_k_binding(dataset, config, fit)
    if config.model_id is ModelID.M8_DUAL_K:
        record = fit.diagnostics["dual_surface_k"]
        model_id = config.model_id.value
        params = dict(record["input_parameters"])
        y0 = float(record["component_baseline"])
        parts = (record["component_1"], record["component_2"])
        labels = tuple(record["component_labels"])
        semantics = str(record["label_semantics"])
        amplitudes = (float(record["signed_B"]), float(record["signed_C"]))
        mapping = tuple(int(i) for i in record["reported_to_input"])
        model = ModelFactory().create(config)

        def evaluate(times):
            total, components = model.evaluate(times, params)
            return total, components[mapping[0]], components[mapping[1]]
    else:
        record = fit.diagnostics.get("analysis_artifacts", {})
        model_id = str(record.get("model_name", ""))
        if model_id not in {"M2_DUAL_HO_D", "M3_DUAL_HO_DK"}:
            raise ValueError("legacy decomposition requires persisted signed contributions and selected model")
        parts = (record.get("component_H", ()), record.get("component_O", ()))
        if "fixed_d" in config.model_specific:
            params = {**fit.diagnostics["fixed_d"]["fixed_parameters"], **fit.params}
            kernel_options = {"term_tol": config.solver.relative_tolerance, "max_terms": config.solver.max_terms}
            y0 = float(params["y0"])
        else:
            adapter, _ = _dual_fit_adapter(config)
            kernel_options = dict(adapter.kernel_options)
            y0 = float(fit.params.get("y0", float(y[0]) if adapter.fixed_baseline is None else adapter.fixed_baseline))
            params = {**fit.params, "y0": y0}
        # Construct actual selected model, preserving geometry/signs but ignoring
        # requested M3 parameters if the original workflow selected M2.
        from pyecr.config import ModelConfig
        actual = ModelConfig(ModelID(model_id), config.model.geometry,
                             surface_mode="finite" if model_id == "M3_DUAL_HO_DK" else "instant",
                             step_type=config.model.step_type, sign_H=config.model.sign_H, sign_O=config.model.sign_O)
        model = ModelFactory().create(actual)
        names = ("D_H_cm2_s", "D_O_cm2_s", "A_H", "A_O", "y0")
        if model_id == "M3_DUAL_HO_DK":
            names += ("k_H_cm_s", "k_O_cm_s")
        params = {n: params[n] for n in names}
        labels = ("Effective H-slot process", "Effective O-slot process")
        semantics = "legacy_declared_H_O_slots_not_species_proof"
        amplitudes = (model.sign_H * params["A_H"], model.sign_O * params["A_O"])

        def evaluate(times):
            result = model.evaluate(times, **params, kernel_options=kernel_options)
            return result.total, result.component_H, result.component_O
    if any(len(p) != len(t) for p in parts):
        raise ValueError("persisted contributions must cover full observations")
    total, c1, c2 = evaluate(t)
    for actual, saved in ((total, fit.y_fit), (c1, parts[0]), (c2, parts[1])):
        _parity(actual, np.asarray(saved), "production observation parity")
    _parity(fit.y_fit, y0 + np.asarray(parts[0]) + parts[1], "zero reconstruction")
    observation = dict(model_id=model_id,
        observation=DecompositionGrid(tuple(t), tuple(fit.y_fit), tuple(parts[0]), tuple(parts[1])),
        observed=tuple(map(float, y)), residuals=tuple(map(float, fit.residuals)),
        original_raw_indices=tuple(map(int, np.flatnonzero(dataset.mask))), y0=y0,
        signed_amplitudes=amplitudes, labels=labels, label_semantics=semantics,
        signal_unit=str(dataset.metadata.get("signal_unit", "arbitrary")),
        signal_type=str(dataset.metadata.get("signal_type", "signal")),
        dataset_sha256=dataset_sha256(dataset), config_sha256=_sha256_json(config.to_dict()),
        fit_result_sha256=_sha256_json(fit.to_dict()), observation_parity_max_abs=float(np.max(np.abs(total-fit.y_fit))))
    return MappingProxyType(observation), evaluate


def build_observation_decomposition(dataset: EcrDataset, config: AnalysisConfig, fit: FitResult):
    """Immutable canonical zero-baseline observations; never builds a display grid.

    This is structural production parity, not a reliability gate. Component
    analysis callers must use the current-parent service gate in P08.
    """
    return _observation_context(dataset, config, fit)[0]


def build_decomposition(dataset: EcrDataset, config: AnalysisConfig, fit: FitResult,
                        options: DecompositionViewOptions | None = None) -> DecompositionData:
    """Existing plotting behavior, sharing the authoritative observation path."""
    options = options or DecompositionViewOptions()
    if not isinstance(options, DecompositionViewOptions):
        raise TypeError("options must be DecompositionViewOptions")
    observation, evaluate = _observation_context(dataset, config, fit)
    t = dataset.processed_time
    parts = (observation['observation'].component_1, observation['observation'].component_2)
    # Include every observation exactly, supplement with deterministic linear and
    # log1p points inside coverage only. Finite kernel mode truncation consequently
    # sees the same earliest positive time as the original fit.
    start, end = float(t[0]), float(t[-1])
    positive = t[t > start] - start
    span = end - start
    scale = float(positive[0]) if len(positive) else 1.
    count = options.display_points // 2
    log_grid = start + np.expm1(np.linspace(0., np.log1p(span / scale), count)) * scale
    grid = np.unique(np.clip(np.r_[t, np.linspace(start, end, count), log_grid], start, end))
    # Do not introduce smaller positive times that could change legacy term
    # selection; retain the initial point and the original first positive time.
    if len(t) > 1 and t[0] == 0:
        grid = grid[(grid == 0) | (grid >= t[1])]
    d_total, d1, d2 = evaluate(grid)
    ix = np.searchsorted(grid, t)
    for actual, saved in ((d_total[ix], fit.y_fit), (d1[ix], parts[0]), (d2[ix], parts[1])):
        _parity(actual, np.asarray(saved), "display observation parity")
    return DecompositionData(**observation, display=DecompositionGrid(tuple(grid), tuple(d_total), tuple(d1), tuple(d2)),
                             options=options)
