"""Explicit prior/iteration evidence; data only, no inference or publication."""
from dataclasses import dataclass
from math import isclose
from typing import ClassVar
from .common import Contract,require,integer,number,vector,digest,strings,text


@dataclass(frozen=True,slots=True,kw_only=True)
class SignedDctPriorSpec(Contract):
    KIND: ClassVar[str]='signed_dct_prior_spec'
    family: str='linear_b1'
    version: str='signed-priors/1'
    alpha1: float=.4
    alpha0: float=2.56
    edge_beta: float=256.
    buffer_decades: float=.5
    hard_zero_control: bool=False
    up_floor: float=.01
    up_slope_ratio: float=5.
    up_curvature_ratio: float=1.
    length_scale_decades: float=.25
    neighbourhood_decades: float=.1
    reference_neighbour_nodes: int=1
    reference_zc_neighbour_nodes: int=0
    up_weight_min: float=.001
    up_weight_max: float=10.
    zc_enabled: bool=True
    zc_strength: float=1.
    zc_floor: float=.1
    zc_slope_ratio: float=1.
    zc_cap_relative_up: float=1.
    adaptive_edge: bool=False
    initial_edge_beta: float=256.
    edge_floor: float=.01
    edge_amplitude_ratio: float=1.
    max_iterations: int=32
    damping: float=.5
    chi_tolerance: float=.001
    prediction_tolerance: float=1e-5
    weight_tolerance: float=.005
    optimality_tolerance: float=1e-8
    stagnation_patience: int=8
    cycle_tolerance: float=1e-5
    initial_density_tolerance: float=.10
    initial_prediction_tolerance: float=.01
    second_initial: str='zero'
    fallback: str='none'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.family in {'linear_b1','up2012_mass_reference','up_density_engineering'},'unknown prior family')
        require(self.version=='signed-priors/1','unknown prior version')
        require(type(self.hard_zero_control) is bool and type(self.zc_enabled) is bool and type(self.adaptive_edge) is bool,'prior switches must be bool')
        for name in ('alpha1','alpha0','edge_beta','initial_edge_beta','up_slope_ratio','up_curvature_ratio','zc_strength','zc_slope_ratio','edge_amplitude_ratio'):
            object.__setattr__(self,name,number(getattr(self,name),name,nonnegative=True))
        for name in ('buffer_decades','up_floor','length_scale_decades','neighbourhood_decades','up_weight_min','up_weight_max',
            'zc_floor','zc_cap_relative_up','edge_floor','damping','chi_tolerance','prediction_tolerance','weight_tolerance',
            'optimality_tolerance','cycle_tolerance','initial_density_tolerance','initial_prediction_tolerance'):
            object.__setattr__(self,name,number(getattr(self,name),name,positive=True))
        require(self.damping<=1 and self.up_weight_min<=self.up_weight_max,'invalid damping/weight limits')
        integer(self.max_iterations,'max_iterations',minimum=2);integer(self.stagnation_patience,'stagnation_patience',minimum=3)
        integer(self.reference_neighbour_nodes,'reference_neighbour_nodes')
        integer(self.reference_zc_neighbour_nodes,'reference_zc_neighbour_nodes')
        require(self.second_initial=='zero' and self.fallback=='none','only declared zero second initial and no implicit fallback are implemented')
        require(not self.hard_zero_control or self.family=='linear_b1','hard zero is a linear sensitivity control only')

    @property
    def adaptive(self):return self.family!='linear_b1'


@dataclass(frozen=True,slots=True,kw_only=True)
class SignedPriorIteration(Contract):
    KIND: ClassVar[str]='signed_prior_iteration'
    initial_label: str
    iteration: int
    fixed_objective_before: float
    fixed_objective_after: float
    weighted_sse: float
    penalties: tuple[tuple[str,float],...]
    chi_change: float
    prediction_change: float
    weight_change: float
    optimality: float
    damping: float
    rank: int
    cutoff: float
    weight_ranges: tuple[tuple[str,float,float],...]
    weights_sha256: str
    seconds: float

    def __post_init__(self):
        Contract.__post_init__(self);text(self.initial_label,'initial_label')
        object.__setattr__(self,'penalties',tuple(tuple(v) for v in self.penalties))
        object.__setattr__(self,'weight_ranges',tuple(tuple(v) for v in self.weight_ranges))
        integer(self.iteration,'iteration',minimum=1);integer(self.rank,'rank',minimum=1)
        for name in ('fixed_objective_before','fixed_objective_after','weighted_sse','chi_change','prediction_change','weight_change','optimality','seconds'):
            number(getattr(self,name),name,nonnegative=True)
        number(self.damping,'damping',positive=True);number(self.cutoff,'cutoff',positive=True)
        require(self.damping<=1,'invalid iteration damping')
        for name,value in self.penalties:text(name,'penalty name');number(value,'penalty',nonnegative=True)
        require(isclose(self.fixed_objective_after,self.weighted_sse+sum(v for _,v in self.penalties),rel_tol=1e-10,abs_tol=1e-13),'iteration objective/parts conflict')
        for name,lo,hi in self.weight_ranges:
            text(name,'weight name');number(lo,'minimum weight',nonnegative=True);number(hi,'maximum weight',nonnegative=True);require(lo<=hi,'weight range conflict')
        digest(self.weights_sha256,'weights_sha256')


@dataclass(frozen=True,slots=True,kw_only=True)
class SignedDctPriorEvidence(Contract):
    KIND: ClassVar[str]='signed_dct_prior_evidence'
    family: str
    status: str
    df_semantics: str
    penalties: tuple[tuple[str,float],...]=()
    free_indices: tuple[int,...]=()
    iteration_trace: tuple[SignedPriorIteration,...]=()
    initial_statuses: tuple[tuple[str,str],...]=()
    initial_density_difference: float | None=None
    initial_prediction_difference: float | None=None
    selected_initial: str | None=None
    regularizer_sha256: str | None=None
    failure_reasons: tuple[str,...]=()
    warning_flags: tuple[str,...]=()
    attempted_chi: tuple[float,...]=()
    attempted_reconstruction: tuple[float,...]=()
    fallback: str='none'

    def __post_init__(self):
        Contract.__post_init__(self)
        for name in ('penalties','initial_statuses'):
            object.__setattr__(self,name,tuple(tuple(v) for v in getattr(self,name)))
        object.__setattr__(self,'free_indices',tuple(self.free_indices))
        object.__setattr__(self,'iteration_trace',tuple(self.iteration_trace))
        require(self.family in {'linear_b1','up2012_mass_reference','up_density_engineering'},'unknown evidence family')
        require(self.status in {'LINEAR_CONVERGED','CONVERGED','FAIL_MAX_ITER','FAIL_CYCLE','FAIL_STALLED','FAIL_INITIAL_DEPENDENCE','FAIL_NUMERIC'},'unknown outer iteration status')
        require(self.df_semantics in {'fixed_linear','conditional_frozen_weights'},'unknown df semantics')
        require(self.df_semantics==('fixed_linear' if self.family=='linear_b1' else 'conditional_frozen_weights'),'family/df semantics conflict')
        require(self.fallback=='none','a failed adaptive solve cannot silently claim fallback success')
        object.__setattr__(self,'failure_reasons',strings(self.failure_reasons,'failure_reasons'))
        object.__setattr__(self,'warning_flags',strings(self.warning_flags,'warning_flags'))
        object.__setattr__(self,'attempted_chi',vector(self.attempted_chi,'attempted_chi'))
        object.__setattr__(self,'attempted_reconstruction',vector(self.attempted_reconstruction,'attempted_reconstruction'))
        for value in self.free_indices:integer(value,'free index')
        require(all(a<b for a,b in zip(self.free_indices,self.free_indices[1:])),'free indices must increase')
        for name,value in self.penalties:text(name,'penalty');number(value,'penalty value',nonnegative=True)
        for value in self.iteration_trace:require(isinstance(value,SignedPriorIteration),'typed iteration trace required')
        for label,status in self.initial_statuses:text(label,'initial');text(status,'initial status')
        for name in ('initial_density_difference','initial_prediction_difference'):
            if getattr(self,name) is not None:number(getattr(self,name),name,nonnegative=True)
        if self.regularizer_sha256 is not None:digest(self.regularizer_sha256,'regularizer_sha256')
        if self.status.startswith('FAIL'):require(bool(self.failure_reasons),'failed outer iteration requires reason')
        else:
            require(not self.failure_reasons and self.selected_initial is not None,'successful prior evidence needs selected initial without hidden failure')
            require(not self.attempted_chi and not self.attempted_reconstruction,'successful arrays belong in the spectrum, not failed-attempt fields')
            if self.family!='linear_b1':require(len(self.initial_statuses)==2 and all(s=='CONVERGED' for _,s in self.initial_statuses),'both declared initializations must converge')
