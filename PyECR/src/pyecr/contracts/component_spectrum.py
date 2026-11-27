"""P09 nonnegative component shape spectra; distinct from signed inversion."""
from dataclasses import dataclass
from math import log10, isclose, exp
from typing import ClassVar

from pyecr.dct.config import DctAnalysisConfig
from .common import Contract, ResultState, require, number, vector, close_vectors
from .dct import DctSourceSnapshot, SourceKind


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentDctConfig(Contract):
    KIND: ClassVar[str] = 'component_dct_config'
    grid: DctAnalysisConfig
    normalization_weight: float = 100.
    minimum_shape_span: float = .01
    minimum_B_over_parent_rmse: float = 3.
    algorithm_version: str = 'component-nonnegative-shared-curvature/1'
    kernel: str = 'exp_decay'
    density_measure: str = 'dlog10_tau'
    regularizer: str = 'sqrt_dlog10_tau_second_derivative_v1'
    solver: str = 'legacy_bounded_trf_unit_mass_v1'
    observation_weighting: str = 'uniform_derived_shape'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.grid,DctAnalysisConfig), 'component grid config required')
        self.grid.validate()
        require(self.grid.lambda_mode == 'manual' and not self.grid.use_classical_bi,
                'component spectra require explicit Manual lambda; no geometry/Bi prerequisite')
        for n in ('normalization_weight','minimum_shape_span','minimum_B_over_parent_rmse'):
            object.__setattr__(self,n,number(getattr(self,n),n,positive=True))
        require(self.minimum_shape_span <= 1 and self.algorithm_version == 'component-nonnegative-shared-curvature/1',
                'unknown component algorithm or invalid information threshold')
        require((self.kernel,self.density_measure,self.regularizer,self.solver,self.observation_weighting)==
            ('exp_decay','dlog10_tau','sqrt_dlog10_tau_second_derivative_v1','legacy_bounded_trf_unit_mass_v1','uniform_derived_shape'),
            'unknown component kernel/measure/regularizer/solver/weighting')


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentSpectrumResult(Contract):
    KIND: ClassVar[str] = 'component_spectrum_result'
    physical_source: DctSourceSnapshot
    source: DctSourceSnapshot
    config: ComponentDctConfig
    state: ResultState
    B: float
    S_total: float
    parent_rmse: float
    tau_s: tuple[float,...]
    log10_tau: tuple[float,...]
    quadrature_weights: tuple[float,...]
    chi: tuple[float,...]
    mass: tuple[float,...]
    reconstruction: tuple[float,...]
    residuals: tuple[float,...]
    common_chi: tuple[float,...]
    common_mass: tuple[float,...]
    optimizer_success: bool
    chi_unit: str = '1/decade'
    mass_unit: str = '1'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.chi_unit=='1/decade' and self.mass_unit=='1','component density/mass unit conflict')
        p,s=self.physical_source,self.source
        require(isinstance(p,DctSourceSnapshot) and isinstance(s,DctSourceSnapshot)
                and isinstance(self.config,ComponentDctConfig) and isinstance(self.state,ResultState),'component result types required')
        p.require_usable();s.require_usable();p.identity.assert_matches(s.identity)
        require(p.source_kind is s.source_kind is SourceKind.FIT_COMPONENT and p.normalization is None
                and s.representation == 'component_remaining' and s.signal_type == 'normalized' and s.signal_unit == '1',
                'physical contribution and independent remaining shape required')
        require(p.component_key == s.component_key and p.time_s == s.time_s and p.t0_s == s.t0_s
                and p.raw_indices == s.raw_indices and p.raw_count == s.raw_count
                and p.component_initial_value == s.component_initial_value and p.component_terminal_value == s.component_terminal_value,
                'component shape source binding conflict')
        for name in ('B','S_total','parent_rmse'):
            object.__setattr__(self,name,number(getattr(self,name),name,positive=name=='S_total',nonnegative=name=='parent_rmse'))
        require(self.B == p.component_initial_value-p.component_terminal_value and self.B != 0,'analytic B conflict')
        for name in ('tau_s','log10_tau','quadrature_weights','chi','mass','reconstruction','residuals','common_chi','common_mass'):
            object.__setattr__(self,name,vector(getattr(self,name),name))
        close_vectors(s.signal,tuple((v-p.component_terminal_value)/self.B for v in p.signal),'remaining shape')
        require(all(-1e-10 <= v <= 1+1e-10 for v in s.signal)
                and all(a >= b-1e-10 for a,b in zip(s.signal,s.signal[1:])), 'shape is not a 1-to-0 monotonic remaining response')
        require(max(s.signal)-min(s.signal) >= self.config.minimum_shape_span
                and abs(self.B) >= self.config.minimum_B_over_parent_rmse*self.parent_rmse,'low-information component')
        n=self.config.grid.n_tau
        require(all(len(getattr(self,k)) == n for k in ('tau_s','log10_tau','quadrature_weights','chi','mass','common_chi','common_mass')),'component grid length conflict')
        require(all(a < b for a,b in zip(self.tau_s,self.tau_s[1:])) and self.tau_s[0]>0,'component tau must increase')
        g=self.config.grid
        require(isclose(self.tau_s[0],g.tau_min_s,rel_tol=1e-12) and isclose(self.tau_s[-1],g.tau_max_s,rel_tol=1e-12),'component tau/config conflict')
        close_vectors(self.log10_tau,tuple(log10(v) for v in self.tau_s),'log10 tau')
        h=(self.log10_tau[-1]-self.log10_tau[0])/(n-1)
        close_vectors(self.log10_tau,tuple(self.log10_tau[0]+i*h for i in range(n)),'uniform log grid')
        close_vectors(self.quadrature_weights,(h/2,*([h]*(n-2)),h/2),'log10 quadrature')
        require(all(v>=0 for v in (*self.chi,*self.mass)),'nonnegative shape spectrum required')
        close_vectors(self.mass,tuple(c*w for c,w in zip(self.chi,self.quadrature_weights)),'density/mass')
        require(isclose(sum(self.mass),1.,abs_tol=1e-10,rel_tol=1e-10),'component shape mass normalization conflict')
        close_vectors(self.common_chi,tuple(self.B/self.S_total*c for c in self.chi),'common signed density')
        close_vectors(self.common_mass,tuple(self.B/self.S_total*c for c in self.mass),'common signed mass')
        require(len(self.reconstruction) == len(self.residuals) == len(s.signal),'component full observation reconstruction required')
        close_vectors(self.reconstruction,tuple(sum(m*exp(-t/tau) for m,tau in zip(self.mass,self.tau_s)) for t in s.time_s),'exponential time reconstruction')
        close_vectors(self.residuals,tuple(a-b for a,b in zip(s.signal,self.reconstruction)),'shape residual')
        require(isinstance(self.optimizer_success,bool) and (self.optimizer_success or not self.state.ready),'failed component optimizer cannot publish ready')
