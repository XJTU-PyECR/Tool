"""Shared versioned grid policy; legacy service imports remain aliases."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Literal, Mapping

@dataclass(frozen=True, slots=True)
class DctAnalysisConfig:
    """Explicit DCT grid and regularization choices supplied by a client."""

    tau_min_s: float = 1.0e-3
    tau_max_s: float = 1.0e3
    n_tau: int = 121
    lambda_mode: Literal["manual", "gcv"] = "gcv"
    regularization_lambda: float | None = None
    use_classical_bi: bool = False
    schema_version: int = 1
    gcv_lambdas: tuple[float, ...] = ()

    def validate(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, Integral) or self.schema_version != 1:
            raise ValueError("unsupported DctAnalysisConfig schema_version")
        tau_min = _positive_finite("tau_min_s", self.tau_min_s)
        tau_max = _positive_finite("tau_max_s", self.tau_max_s)
        if tau_min >= tau_max:
            raise ValueError("tau_min_s must be smaller than tau_max_s")
        if isinstance(self.n_tau, bool) or not isinstance(self.n_tau, Integral):
            raise TypeError("n_tau must be an integer")
        if self.n_tau < 10:
            raise ValueError("n_tau must be at least 10")
        if self.lambda_mode not in {"manual", "gcv"}:
            raise ValueError("lambda_mode must be 'manual' or 'gcv'")
        if self.lambda_mode == "manual":
            if self.regularization_lambda is None:
                raise ValueError("manual lambda mode requires regularization_lambda")
            _nonnegative_finite(
                "regularization_lambda", self.regularization_lambda
            )
        elif self.regularization_lambda is not None:
            raise ValueError("GCV lambda mode must not set regularization_lambda")
        if not isinstance(self.use_classical_bi, bool):
            raise TypeError("use_classical_bi must be bool")
        if self.gcv_lambdas:
            if self.lambda_mode != 'gcv':
                raise ValueError('Manual lambda must not carry GCV candidates')
            if len(self.gcv_lambdas) < 3:
                raise ValueError('GCV requires at least three candidates')
            for value in self.gcv_lambdas:
                _positive_finite('gcv_lambdas', value)
            if any(a >= b for a, b in zip(self.gcv_lambdas, self.gcv_lambdas[1:])):
                raise ValueError('GCV candidates must be strictly increasing')

    def to_dict(self) -> dict[str, object]:
        self.validate()
        value = {
            "tau_min_s": float(self.tau_min_s),
            "tau_max_s": float(self.tau_max_s),
            "n_tau": int(self.n_tau),
            "lambda_mode": self.lambda_mode,
            "regularization_lambda": self.regularization_lambda,
            "use_classical_bi": self.use_classical_bi,
            "schema_version": int(self.schema_version),
        }
        # Omit the optional extension for byte-compatible historical defaults.
        if self.gcv_lambdas:
            value['gcv_lambdas'] = list(self.gcv_lambdas)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DctAnalysisConfig:
        config = cls(
            schema_version=value.get("schema_version", 1),
            tau_min_s=float(value["tau_min_s"]),
            tau_max_s=float(value["tau_max_s"]),
            n_tau=int(value.get("n_tau", 121)),
            lambda_mode=str(value.get("lambda_mode", "gcv")),  # type: ignore[arg-type]
            regularization_lambda=(
                None
                if value.get("regularization_lambda") is None
                else float(value["regularization_lambda"])
            ),
            use_classical_bi=bool(value.get("use_classical_bi", False)),
            gcv_lambdas=tuple(float(x) for x in value.get('gcv_lambdas', ())),
        )
        config.validate()
        return config


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: Real) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(name: str, value: Real) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


