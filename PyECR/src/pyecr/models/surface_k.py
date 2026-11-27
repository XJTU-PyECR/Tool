"""Analytic lumped surface exchange; independent of diffusion series kernels."""
from dataclasses import dataclass
from math import isfinite, log
from numbers import Real
from typing import Mapping
import numpy as np
from pyecr.config import SampleGeometry
from pyecr.models.ids import ModelID

MODEL_VERSION = "surface-k/1.0.0"
CONTRACT_VERSION = "pyecr-k-only-contract/1.0.0"
EXPOSURE_RULE = "active_axes_both_faces_uniform_k_v1"

def finite_scalar(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return value

@dataclass(frozen=True, slots=True)
class SurfaceKKernel:
    geometry: SampleGeometry

    def __post_init__(self):
        if not isinstance(self.geometry, SampleGeometry):
            raise TypeError("geometry must be SampleGeometry")
        self.geometry.validate()
        self.g_cm_inv  # validate representable geometry before evaluation

    @property
    def g_cm_inv(self):
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            g = float(np.sum([2.0 / np.float64(d) for d, active in zip(
                self.geometry.full_dims_cm, self.geometry.active_dimensions, strict=True) if active]))
        if not isfinite(g) or g <= 0:
            raise ValueError("NUMERIC_RANGE_UNSUPPORTED: g is not positive finite float64")
        return g

    def rates(self, k_cm_s):
        k = finite_scalar(k_cm_s, "k_cm_s", True)
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            rate = float(np.float64(k) * self.g_cm_inv)
            tau = float(np.float64(1.0) / rate)
        if not isfinite(rate) or rate <= 0 or not isfinite(tau) or tau <= 0:
            raise ValueError("NUMERIC_RANGE_UNSUPPORTED: lambda/tau must be positive finite float64")
        return rate, tau

    def evaluate(self, t_s, k_cm_s):
        """Return completion X and dX/dlog10(k), with exact t=0 limits."""
        times = np.asarray(t_s, dtype=np.float64)
        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("time must be a finite nonnegative vector in seconds")
        rate, _tau = self.rates(k_cm_s)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            z = rate * times
            x = -np.expm1(-z)
            gradient = np.zeros_like(z)
            finite = np.isfinite(z)
            gradient[finite] = log(10.0) * z[finite] * np.exp(-z[finite])
        x[times == 0] = 0.0
        x.flags.writeable = False
        gradient.flags.writeable = False
        return x, gradient

@dataclass(frozen=True, slots=True)
class SurfaceKModel:
    kernel: SurfaceKKernel
    model_id: ModelID = ModelID.M7_SINGLE_K

    def __post_init__(self):
        if not isinstance(self.kernel, SurfaceKKernel) or self.model_id is not ModelID.M7_SINGLE_K:
            raise TypeError("SurfaceKModel requires its independent SurfaceKKernel and M7 ID")

    def predict(self, t, theta: Mapping[str, float], context=None):
        if not isinstance(theta, Mapping):
            raise TypeError("theta must be a physical parameter mapping")
        if set(theta) - {"k_cm_s", "amplitude", "baseline"}:
            raise ValueError("M7 accepts k/amplitude/baseline only; no D")
        k = theta["k_cm_s"]
        amplitude = finite_scalar(theta.get("amplitude", 1.0), "amplitude")
        baseline = finite_scalar(theta.get("baseline", 0.0), "baseline")
        x, _ = self.kernel.evaluate(t, k)
        with np.errstate(over="ignore", invalid="ignore"):
            y = baseline + amplitude * x
        if not np.all(np.isfinite(y)):
            raise ValueError("NUMERIC_RANGE_UNSUPPORTED: nonfinite observed prediction")
        y.flags.writeable = False
        return y

    def jacobian(self, t, theta, free_parameters):
        x, dk = self.kernel.evaluate(t, theta["k_cm_s"])
        columns = {"k_cm_s": theta.get("amplitude", 1.0) * dk,
                   "amplitude": x, "baseline": np.ones_like(x)}
        result = np.column_stack([columns[p] for p in free_parameters])
        if not np.all(np.isfinite(result)):
            raise ValueError("NUMERIC_RANGE_UNSUPPORTED: nonfinite Jacobian")
        return result
