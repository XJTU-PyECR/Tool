"""Reusable M2/M3 physical-parameter objective for the staged H/O fitter.

The sole scientific forward operation is DualHOModel.evaluate().  Optimizer
residuals are prediction minus observation; reported residuals are the reverse.
Raw SSE is never replaced by SciPy's optional robust optimizer cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyecr.fitting.base import FitResult
from pyecr.models.dual_ho import DualHOModel, StepType
from pyecr.models.ids import ModelID
from pyecr.fitting.progress import count_fit_call

if TYPE_CHECKING:
    from pyecr.config import SampleGeometry
    from pyecr.fitting.dual_ho import DualHOFitConfig, _Parameterization
    from pyecr.services.workflow import AnalysisSession, WorkflowRevision

__all__ = [
    "DualHOObjectiveContext", "DualHOObjectiveEvaluation",
    "DualHOObjectiveEvaluator", "DualObjectiveRevision", "OptimizerLossMetadata",
]


@dataclass(frozen=True, slots=True)
class DualObjectiveRevision:
    """Source revisions from the current single-dataset workflow."""

    dataset_revision: int
    preprocessing_revision: int
    model_revision: int
    fit_revision: WorkflowRevision


@dataclass(frozen=True, slots=True)
class OptimizerLossMetadata:
    """Optimizer loss identity; not the reported raw-SSE definition."""

    name: str
    f_scale: float = 1.0
    raw_sse_is_optimizer_cost: bool = False
    linear_cost_is_half_raw_sse: bool = False

    def __post_init__(self) -> None:
        if self.name not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("unknown dual H/O optimizer loss")
        if self.f_scale != 1.0:
            raise ValueError("dual H/O fitter currently uses scipy's fixed f_scale=1.0")
        object.__setattr__(self, "raw_sse_is_optimizer_cost", False)
        object.__setattr__(self, "linear_cost_is_half_raw_sse", self.name == "linear")


@dataclass(frozen=True, slots=True)
class DualHOObjectiveContext:
    """Captured inputs and conventions for one M2 or M3 objective.

    Legacy direct ``fit_dual_ho(t, y, model)`` calls lack a source dataset and
    workflow; for those calls geometry/source units/revisions are explicitly
    ``None``. Production ``from_fit_result`` requires all of them.
    """

    model: DualHOModel
    model_id: ModelID
    processed_time_s: NDArray[np.float64]
    processed_signal: NDArray[np.float64]
    full_dims_cm: tuple[float, float, float] | None
    active_dimensions: tuple[bool, bool, bool] | None
    signal_type: str | None
    signal_unit: str | None
    step_type: StepType
    sign_H: float
    sign_O: float
    kernel_options: Mapping[str, object]
    parameter_bounds: Mapping[str, tuple[float, float]]
    fixed_baseline: float | None
    label_constraint: str
    optimizer_loss: OptimizerLossMetadata
    revision: DualObjectiveRevision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, DualHOModel):
            raise TypeError("model must be DualHOModel")
        if self.model_id not in {ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK}:
            raise ValueError("dual objective requires M2 or M3")
        times = _readonly_vector(self.processed_time_s, "processed_time_s")
        signal = _readonly_vector(self.processed_signal, "processed_signal")
        if times.size != signal.size or times.size < 3:
            raise ValueError("processed arrays must align and contain at least three points")
        if np.any(times < 0.0) or not np.all(np.diff(times) > 0.0):
            raise ValueError("processed time must be nonnegative and strictly increasing")
        if self.step_type is not self.model.step_type or self.sign_H != self.model.sign_H or self.sign_O != self.model.sign_O:
            raise ValueError("signal convention must match DualHOModel")
        if self.label_constraint not in {"D_H>=D_O", "none"}:
            raise ValueError("label_constraint must be D_H>=D_O or none")
        if (self.full_dims_cm is None) != (self.active_dimensions is None):
            raise ValueError("geometry dimensions and active flags must be supplied together")
        if self.full_dims_cm is not None:
            from pyecr.config import SampleGeometry
            SampleGeometry(self.full_dims_cm, self.active_dimensions).validate()
            kernel = self.model.kernel
            if (hasattr(kernel, "full_dims_cm") and
                    tuple(kernel.full_dims_cm) != tuple(self.full_dims_cm)):
                raise ValueError("objective geometry differs from model kernel")
            if (hasattr(kernel, "active_dimensions") and
                    tuple(kernel.active_dimensions) != tuple(self.active_dimensions)):
                raise ValueError("objective active dimensions differ from model kernel")
        bounds: dict[str, tuple[float, float]] = {}
        for name, pair in self.parameter_bounds.items():
            if len(pair) != 2 or not all(isfinite(float(value)) for value in pair):
                raise ValueError(f"{name} bounds must contain two finite values")
            lo, hi = float(pair[0]), float(pair[1])
            if lo > hi:
                raise ValueError(f"{name} bounds are reversed")
            bounds[name] = (lo, hi)
        required = {"D_H_cm2_s", "D_O_cm2_s", "A_H", "A_O"}
        if self.model_id is ModelID.M3_DUAL_HO_DK:
            required.update({"k_H_cm_s", "k_O_cm_s"})
        if not required.issubset(bounds):
            raise ValueError("dual objective is missing kinetic/amplitude bounds")
        if self.fixed_baseline is not None and not isfinite(self.fixed_baseline):
            raise ValueError("fixed_baseline must be finite")
        if self.fixed_baseline is None and "y0" not in bounds:
            raise ValueError("free baseline requires y0 bounds")
        if not isinstance(self.optimizer_loss, OptimizerLossMetadata):
            raise TypeError("optimizer_loss must be OptimizerLossMetadata")
        if self.revision is not None:
            for name in ("dataset_revision", "preprocessing_revision", "model_revision"):
                value = getattr(self.revision, name)
                if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                    raise ValueError(f"{name} must be a nonnegative integer")
        object.__setattr__(self, "processed_time_s", times)
        object.__setattr__(self, "processed_signal", signal)
        object.__setattr__(self, "kernel_options", MappingProxyType(dict(self.kernel_options)))
        object.__setattr__(self, "parameter_bounds", MappingProxyType(bounds))


@dataclass(frozen=True, slots=True)
class DualHOObjectiveEvaluation:
    prediction: NDArray[np.float64]
    residuals: NDArray[np.float64]
    optimizer_residuals: NDArray[np.float64]
    raw_sse: float


@dataclass(frozen=True, slots=True)
class DualHOObjectiveEvaluator:
    context: DualHOObjectiveContext

    @property
    def optimizer_loss(self) -> OptimizerLossMetadata:
        return self.context.optimizer_loss

    def predict(self, params: Mapping[str, float]) -> NDArray[np.float64]:
        count_fit_call("forward")
        values = self._validated_params(params)
        is_m3 = self.context.model_id is ModelID.M3_DUAL_HO_DK
        prediction = np.asarray(self.context.model.evaluate(
            self.context.processed_time_s,
            D_H_cm2_s=values["D_H_cm2_s"], D_O_cm2_s=values["D_O_cm2_s"],
            A_H=values["A_H"], A_O=values["A_O"], y0=values["y0"],
            k_H_cm_s=values.get("k_H_cm_s") if is_m3 else None,
            k_O_cm_s=values.get("k_O_cm_s") if is_m3 else None,
            kernel_options=self.context.kernel_options,
        ).total, dtype=np.float64)
        if prediction.shape != self.context.processed_signal.shape or not np.all(np.isfinite(prediction)):
            raise FloatingPointError("dual H/O prediction is nonfinite or misaligned")
        return prediction

    def evaluate(self, params: Mapping[str, float]) -> DualHOObjectiveEvaluation:
        prediction = self.predict(params)
        observed = self.context.processed_signal
        residuals = observed - prediction
        optimizer_residuals = prediction - observed
        return DualHOObjectiveEvaluation(
            prediction=prediction, residuals=residuals,
            optimizer_residuals=optimizer_residuals,
            raw_sse=float(np.dot(residuals, residuals)),
        )

    def evaluate_residual(self, params: Mapping[str, float]) -> NDArray[np.float64]:
        """Return the exact prediction-minus-observation optimizer residual."""

        count_fit_call("residual")
        return self.predict(params) - self.context.processed_signal

    def evaluate_raw_sse(self, params: Mapping[str, float]) -> float:
        """Return unweighted raw SSE, independent of robust optimizer loss."""

        return self.evaluate(params).raw_sse

    def _validated_params(self, params: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(params, Mapping):
            raise TypeError("dual objective params must be a mapping")
        required = {"D_H_cm2_s", "D_O_cm2_s", "A_H", "A_O"}
        if self.context.model_id is ModelID.M3_DUAL_HO_DK:
            required.update({"k_H_cm_s", "k_O_cm_s"})
        missing = sorted(required - set(params))
        if missing:
            raise ValueError("dual objective is missing " + ", ".join(missing))
        if set(params) - (required | {"y0"}):
            raise ValueError("dual objective received unexpected parameters")
        values = {name: float(value) for name, value in params.items()}
        if self.context.fixed_baseline is not None:
            supplied = values.get("y0")
            if supplied is not None and supplied != self.context.fixed_baseline:
                raise ValueError("y0 differs from the fixed baseline")
            values["y0"] = self.context.fixed_baseline
        elif "y0" not in values:
            raise ValueError("free baseline y0 is missing")
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            bounds = self.context.parameter_bounds.get(name)
            if bounds is not None:
                lower_tolerance = 1e-12 * max(abs(value), abs(bounds[0]), 1e-300)
                upper_tolerance = 1e-12 * max(abs(value), abs(bounds[1]), 1e-300)
                if not (bounds[0] - lower_tolerance <= value <= bounds[1] + upper_tolerance):
                    raise ValueError(f"{name} lies outside validated bounds")
            if (name.startswith(("D_", "k_")) and value <= 0.0) or (name.startswith("A_") and value < 0.0):
                raise ValueError(f"{name} has an invalid physical sign")
        if (self.context.label_constraint == "D_H>=D_O" and
                values["D_H_cm2_s"] < values["D_O_cm2_s"]):
            raise ValueError("label constraint D_H>=D_O was violated")
        return values

    @classmethod
    def from_fit_result(
        cls, fit_result: FitResult, session: AnalysisSession,
    ) -> DualHOObjectiveEvaluator:
        """Capture only a current, actually selected production M3 result."""

        from pyecr.fitting.dual_ho import _Parameterization
        from pyecr.models.factory import ModelFactory
        from pyecr.services.analysis import _dual_fit_adapter
        from pyecr.services.workflow import AnalysisSession

        if not isinstance(fit_result, FitResult) or not isinstance(session, AnalysisSession):
            raise TypeError("from_fit_result requires FitResult and AnalysisSession")
        dataset, config = session.dataset, session.current_analysis_config
        if dataset is None or config is None or session.fit_result is not fit_result:
            raise ValueError("FitResult is not the current session fit")
        if (session.fit_stale or session.fit_revision is None or
                session.fit_revision != session.current_revision):
            raise ValueError("FitResult revision is Out of date")
        if config.model_id is not ModelID.M3_DUAL_HO_DK:
            raise ValueError("from_fit_result requires M3 configuration")
        dispatch = fit_result.diagnostics.get("analysis_dispatch")
        if (not isinstance(dispatch, Mapping) or
                dispatch.get("selected_model_id") != ModelID.M3_DUAL_HO_DK.value or
                dispatch.get("m3_ran") is not True):
            raise ValueError("final selected model must be M3, not an M2 fallback")
        fit_config, _ = _dual_fit_adapter(config)
        parameterization = _Parameterization(fit_config, dataset.processed_signal, is_m3=True)
        model = ModelFactory().create(config)
        if not isinstance(model, DualHOModel):
            raise TypeError("factory did not construct the M3 DualHOModel")
        revision = DualObjectiveRevision(
            dataset_revision=session.data_revision,
            preprocessing_revision=session.preprocessing_revision,
            model_revision=session.config_revision,
            fit_revision=session.fit_revision,
        )
        context = _context_for_level(
            model, dataset.processed_time, dataset.processed_signal,
            parameterization=parameterization, fit_config=fit_config,
            is_m3=True, geometry=config.model.geometry,
            signal_type=str(dataset.metadata.get("signal_type", "unknown")),
            signal_unit=str(dataset.metadata.get("signal_unit", "unspecified")),
            revision=revision,
        )
        evaluator = cls(context)
        actual_sse = evaluator.evaluate_raw_sse(fit_result.params)
        if not isclose(actual_sse, fit_result.SSE, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(
                "M3 fitted-point raw SSE does not match FitResult.SSE: "
                f"evaluated={actual_sse:.17g}, reported={fit_result.SSE:.17g}"
            )
        return evaluator


def _context_for_level(
    model: DualHOModel,
    times: NDArray[np.float64],
    signal: NDArray[np.float64],
    *,
    parameterization: _Parameterization,
    fit_config: DualHOFitConfig,
    is_m3: bool,
    geometry: SampleGeometry | None = None,
    signal_type: str | None = None,
    signal_unit: str | None = None,
    revision: DualObjectiveRevision | None = None,
) -> DualHOObjectiveContext:
    """Adapt the existing fitter layout; no H/O equation is implemented here."""

    kernel = model.kernel
    if geometry is not None:
        full_dims = tuple(geometry.full_dims_cm)
        active = tuple(geometry.active_dimensions)
    elif hasattr(kernel, "full_dims_cm") and hasattr(kernel, "active_dimensions"):
        full_dims = tuple(kernel.full_dims_cm)
        active = tuple(kernel.active_dimensions)
    else:
        # The legacy low-level API accepts user kernels without geometry
        # attributes. Provenance is absent rather than partially invented.
        full_dims = None
        active = None
    return DualHOObjectiveContext(
        model=model, model_id=ModelID.M3_DUAL_HO_DK if is_m3 else ModelID.M2_DUAL_HO_D,
        processed_time_s=times, processed_signal=signal,
        full_dims_cm=full_dims, active_dimensions=active,
        signal_type=signal_type, signal_unit=signal_unit,
        step_type=model.step_type, sign_H=model.sign_H, sign_O=model.sign_O,
        kernel_options=fit_config.kernel_options,
        parameter_bounds=parameterization.physical_bounds,
        fixed_baseline=None if fit_config.fit_baseline else parameterization.fixed_baseline,
        label_constraint=fit_config.label_constraint.value,
        optimizer_loss=OptimizerLossMetadata(fit_config.loss),
        revision=revision,
    )


def _readonly_vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.flags.writeable = False
    return result
