"""Immutable scientific arrays shared by GUI and static renderers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

__all__ = ["FitPlotData"]


@dataclass(frozen=True, slots=True)
class FitPlotData:
    """Detached values needed for fit and residual visualizations."""

    time_s: tuple[float, ...]
    observed: tuple[float, ...]
    fitted: tuple[float, ...]
    residuals: tuple[float, ...]
    component_H: tuple[float, ...] = ()
    component_O: tuple[float, ...] = ()
    model_name: str = "unknown"
    signal_label: str = "Signal"
    warnings: tuple[str, ...] = ()
    component_1: tuple[float,...] = ()
    component_2: tuple[float,...] = ()
    component_labels: tuple[str,...] = ()
    component_baseline: float = 0.

    def __post_init__(self) -> None:
        for name in (
            "time_s",
            "observed",
            "fitted",
            "residuals",
            "component_H",
            "component_O",
            "component_1", "component_2",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not all(isfinite(value) for value in values):
                raise ValueError(f"{name} values must be finite")
            object.__setattr__(self, name, values)
        size = len(self.time_s)
        if size == 0:
            raise ValueError("plot data must contain observations")
        if any(
            len(values) != size
            for values in (self.observed, self.fitted, self.residuals)
        ):
            raise ValueError("time, observed, fitted, and residual arrays must align")
        for name in ("component_H", "component_O", "component_1", "component_2"):
            values = getattr(self, name)
            if values and len(values) != size:
                raise ValueError(f"{name} must be empty or align with time_s")
        if any(right <= left for left, right in zip(self.time_s, self.time_s[1:])):
            raise ValueError("time_s must be strictly increasing")
        object.__setattr__(self, "model_name", str(self.model_name))
        object.__setattr__(self, "signal_label", str(self.signal_label))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @property
    def has_components(self) -> bool:
        """Only dual-process models may expose H/O contribution curves."""

        if self.model_name=="M8_DUAL_K": return bool(self.component_1 and self.component_2)
        return self.model_name in {"M2_DUAL_HO_D", "M3_DUAL_HO_DK"} and bool(
            self.component_H or self.component_O
        )

    @classmethod
    def from_fit_presentation(cls, presentation: object) -> FitPlotData:
        """Adapt the UI presentation without importing GUI or service code."""

        return cls(
            time_s=tuple(getattr(presentation, "time_s")),
            observed=tuple(getattr(presentation, "data")),
            fitted=tuple(getattr(presentation, "fit")),
            residuals=tuple(getattr(presentation, "residuals")),
            component_H=tuple(getattr(presentation, "component_H", ())),
            component_O=tuple(getattr(presentation, "component_O", ())),
            model_name=str(getattr(presentation, "model_name", "unknown")),
            signal_label="Signal",
            warnings=tuple(getattr(presentation, "warnings", ())),
            component_1=tuple(getattr(presentation,"component_1",())),
            component_2=tuple(getattr(presentation,"component_2",())),
            component_labels=tuple(getattr(presentation,"component_labels",())),
            component_baseline=float(getattr(presentation,"component_baseline",0.)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **({"component_1":list(self.component_1),"component_2":list(self.component_2),
                "component_labels":list(self.component_labels),"component_baseline":self.component_baseline} if self.component_1 else {}),
            "time_s": list(self.time_s),
            "observed": list(self.observed),
            "fitted": list(self.fitted),
            "residuals": list(self.residuals),
            "component_H": list(self.component_H),
            "component_O": list(self.component_O),
            "model_name": self.model_name,
            "signal_label": self.signal_label,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FitPlotData:
        return cls(
            time_s=_float_tuple(value, "time_s"),
            observed=_float_tuple(value, "observed"),
            fitted=_float_tuple(value, "fitted"),
            residuals=_float_tuple(value, "residuals"),
            component_H=_float_tuple(value, "component_H", required=False),
            component_O=_float_tuple(value, "component_O", required=False),
            model_name=str(value.get("model_name", "unknown")),
            signal_label=str(value.get("signal_label", "Signal")),
            warnings=_string_tuple(value.get("warnings", ()), "warnings"),
            component_1=_float_tuple(value,"component_1",required=False),
            component_2=_float_tuple(value,"component_2",required=False),
            component_labels=tuple(value.get("component_labels",())),
            component_baseline=float(value.get("component_baseline",0.)),
        )


def _float_tuple(
    value: Mapping[str, object],
    name: str,
    *,
    required: bool = True,
) -> tuple[float, ...]:
    raw = value.get(name)
    if raw is None and not required:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return tuple(float(item) for item in raw)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return tuple(str(item) for item in value)
