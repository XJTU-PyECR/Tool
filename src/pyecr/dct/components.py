"""Normalize model-derived shapes and reuse the established bounded solver."""
from dataclasses import replace
import numpy as np

from pyecr.contracts import ComponentDctConfig,ComponentSpectrumResult,ResultState,SignedDctConfig
from pyecr.contracts.common import require,number
from pyecr.dct.core import _solve_spectrum
from pyecr.dct.signed import _operators
from pyecr.dct.execution import isolated_numeric


@isolated_numeric('component')
def fit_component_spectrum(physical_source, parent_source, config:ComponentDctConfig, *, parent_rmse):
    """Explicit P09 operation on a frozen copy, not an Analysis preprocessing step.

    Reuses the old bounded TRF solve and its soft unit-mass row/post-normalization
    for a 1-to-0 component shape only. Uses the P04 grid and scaled L, not the old
    unscaled curvature. No GCV or noise independence/CI claim for derived shapes.
    """
    physical_source.require_usable();parent_source.require_usable()
    require(isinstance(config,ComponentDctConfig),'component config required')
    p=physical_source
    p.identity.assert_matches(parent_source.identity)
    require(parent_source.normalization is not None and p.time_s==parent_source.time_s
            and p.raw_indices==parent_source.raw_indices and p.raw_count==parent_source.raw_count
            and p.t0_s==parent_source.t0_s,'component/whole observation or normalization conflict')
    require(p.signal_unit==parent_source.normalization.spec.signal_unit,'component/whole physical unit conflict')
    parent_rmse=number(parent_rmse,'parent_rmse',nonnegative=True)
    B=p.component_initial_value-p.component_terminal_value
    S=parent_source.normalization.scale_value
    floor=64*np.finfo(float).eps*max(abs(p.component_initial_value),abs(p.component_terminal_value),
        float(np.max(np.abs(p.signal))),np.finfo(float).tiny)
    require(abs(B)>floor,'COMPONENT_ZERO_B: analytic endpoint difference is at roundoff')
    require(abs(B)>=config.minimum_B_over_parent_rmse*parent_rmse,
            'COMPONENT_LOW_INFORMATION_B: below declared multiple of parent residual RMSE; heuristic, not independent component noise')
    r=(np.asarray(p.signal)-p.component_terminal_value)/B
    require(np.ptp(r)>=config.minimum_shape_span,'COMPONENT_LOW_INFORMATION_COVERAGE: observed remaining shape span is too small')
    require(np.all(r>=-1e-10) and np.all(r<=1+1e-10) and np.all(np.diff(r)<=1e-10),
            'COMPONENT_SHAPE_UNSUPPORTED: remaining shape is not monotonic in [0,1]; no clipping or smoothing')
    source=replace(p,signal=tuple(r),signal_type='normalized',signal_unit='1',representation='component_remaining',
        parent_source_sha256=parent_source.content_sha256,normalization=None)
    tau,x,q,A,L,h=_operators(source.time_s,SignedDctConfig(grid=config.grid))
    K=A/q
    solution=_solve_spectrum(K,r,1-r,q,L/q,regularization_lambda=config.grid.regularization_lambda,
                             normalization_weight=config.normalization_weight)
    require(solution.success,'COMPONENT_OPTIMIZER_FAILED: bounded solver did not declare success')
    # Stored remaining prediction is computed on every original observation.
    reconstruction=K@solution.mass
    return ComponentSpectrumResult(physical_source=p,source=source,config=config,
        state=ResultState(status='warning',warnings=('MODEL_DERIVED_SHAPE: no independent experimental/noise evidence or calibrated CI',
            'MANUAL_COMPONENT_LAMBDA: shared P04 grid/curvature; unit mass belongs to component shape only')),
        B=B,S_total=S,parent_rmse=parent_rmse,tau_s=tuple(tau),log10_tau=tuple(x),quadrature_weights=tuple(q),
        chi=tuple(solution.chi),mass=tuple(solution.mass),reconstruction=tuple(reconstruction),residuals=tuple(r-reconstruction),
        common_chi=tuple(B/S*solution.chi),common_mass=tuple(B/S*solution.mass),optimizer_success=solution.success)
