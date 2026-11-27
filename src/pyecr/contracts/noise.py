"""R10 immutable noise provenance and descriptive post-fit evidence.

These extend the existing Contract/SourceIdentity protocol, not FitResult.
No stopping, reliability classification or probability calculation lives here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from math import isclose,sqrt
from .common import Contract,SourceIdentity,require,number,integer,text,strings,vector,rows,digest


@dataclass(frozen=True,slots=True,kw_only=True)
class NoiseReference(Contract):
    KIND:ClassVar[str]='noise_reference'
    source:SourceIdentity
    origin:SourceIdentity
    kind:str
    sigma:float
    signal_type:str
    signal_unit:str
    response_profile:str
    assumptions:str
    basis:str
    data_origin:str
    method:str
    raw_count:int
    time_unit:str
    raw_indices:tuple[int,...]=()
    sample_times:tuple[float,...]=()
    sample_values:tuple[float,...]=()
    interval:tuple[float,float]|None=None
    sample_count:int|None=None
    input_sigma:float=0.
    input_unit:str=''
    input_profile:str=''
    parent_reference_id:str|None=None
    normalization_scale:float|None=None
    normalization_scale_unit:str|None=None
    normalization_version:str|None=None
    issues:tuple[str,...]=()
    conditions:tuple[str,...]=()

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.source,SourceIdentity) and isinstance(self.origin,SourceIdentity),'Noise needs actual dataset/recipe identities')
        require(self.source.fit_id is None and self.origin.fit_id is None,'Noise cannot be estimated from current fitted residuals')
        require(self.kind in {'manual_sd','accepted_initial_platform','accepted_final_platform'},'Unknown noise source kind')
        for k in ('sigma','input_sigma'):object.__setattr__(self,k,number(getattr(self,k),k,positive=True))
        for k in ('signal_type','signal_unit','response_profile','basis','data_origin','method','time_unit','input_unit','input_profile'):text(getattr(self,k),k)
        from pyecr.units import validate_pair,factor_to_reference
        validate_pair(self.signal_type,self.signal_unit)
        require(self.assumptions in {'unknown','iid_stationary_constant_sd','correlated','heteroscedastic'},'Unknown noise assumptions')
        integer(self.raw_count,'raw_count',minimum=1)
        object.__setattr__(self,'raw_indices',rows(self.raw_indices,self.raw_count))
        for k in ('sample_times','sample_values'):object.__setattr__(self,k,vector(getattr(self,k),k))
        for k in ('issues','conditions'):object.__setattr__(self,k,strings(getattr(self,k),k))
        if self.kind=='manual_sd':
            require(self.sample_count is None and self.interval is None and not self.raw_indices and not self.sample_values and not self.sample_times,
                    'External manual SD has no invented sample size/window/rows')
            require(self.method=='external_declared_sd','Unknown manual noise method')
        else:
            n=integer(self.sample_count,'sample_count',minimum=2)
            require(n==len(self.raw_indices)==len(self.sample_values)==len(self.sample_times),'Noise platform row mapping conflict')
            require(isinstance(self.interval,(tuple,list)) and len(self.interval)==2,'Noise platform interval required')
            object.__setattr__(self,'interval',tuple(number(x,'interval') for x in self.interval))
            require(self.interval[0]<=self.interval[1] and all(self.interval[0]<=x<=self.interval[1] for x in self.sample_times),'Platform time interval conflict')
            require(self.method=='accepted_window_sample_sd_ddof1','Unknown platform noise method')
            mean=sum(self.sample_values)/n
            sd=sqrt(sum((x-mean)**2 for x in self.sample_values)/(n-1))
            require(isclose(sd,self.input_sigma,rel_tol=1e-10,abs_tol=1e-14),'Noise input SD conflicts with saved platform values')
        if self.parent_reference_id is None:
            require(self.normalization_scale is self.normalization_scale_unit is self.normalization_version is None,'Normalization provenance missing parent')
            require((self.kind!='manual_sd' or self.origin==self.source) and self.sigma==self.input_sigma and self.signal_unit==self.input_unit and self.response_profile==self.input_profile,
                    'Unconverted noise source/scale conflict')
        else:
            digest(self.parent_reference_id,'parent_reference_id')
            number(self.normalization_scale,'normalization_scale',positive=True)
            text(self.normalization_scale_unit,'normalization_scale_unit');text(self.normalization_version,'normalization_version')
            require(self.signal_type=='normalized' and self.signal_unit=='1','Noise conversion requires normalized output')
            expected=self.input_sigma*factor_to_reference(self.input_unit)/factor_to_reference(self.normalization_scale_unit)/self.normalization_scale
            require(isclose(self.sigma,expected,rel_tol=1e-12,abs_tol=1e-14),'Converted noise SD/accepted scale conflict')


@dataclass(frozen=True,slots=True,kw_only=True)
class NoiseComparison(Contract):
    KIND:ClassVar[str]='noise_comparison'
    source:SourceIdentity
    reference:NoiseReference|None
    residual_role:str
    raw_indices:tuple[int,...]
    raw_count:int
    residuals:tuple[float,...]
    time:tuple[float,...]
    unit:str
    response_profile:str
    p_free:int
    raw_SSE:float
    raw_RMSE:float
    RSE:float|None
    sigma_noise:float|None
    RSE_over_sigma:float|None
    noise_unit_factor:float|None
    status:str
    reasons:tuple[str,...]
    warnings:tuple[str,...]
    conditions:tuple[str,...]
    loss:str
    weighting:str
    rank:int|None
    lag1:float|None=None
    drift_span_over_sd:float|None=None
    half_sd_ratio:float|None=None
    execution_request_id:str|None=None
    N:int=0
    df:int=0

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.source,SourceIdentity),'Noise comparison needs a source')
        require(self.reference is None or isinstance(self.reference,NoiseReference),'Invalid noise reference')
        require(self.residual_role in {'physical_fit_residual','stopped_native_candidate'},'Unknown residual role')
        require(self.residual_role!='physical_fit_residual' or self.source.fit_id is not None,'Fit residual requires a real parent Fit')
        if self.execution_request_id is not None:digest(self.execution_request_id,'execution_request_id')
        for k in ('unit','response_profile','loss','weighting'):text(getattr(self,k),k)
        for k in ('reasons','warnings','conditions'):object.__setattr__(self,k,strings(getattr(self,k),k))
        for k in ('residuals','time'):object.__setattr__(self,k,vector(getattr(self,k),k))
        integer(self.raw_count,'raw_count',minimum=1);object.__setattr__(self,'raw_indices',rows(self.raw_indices,self.raw_count))
        integer(self.N,'N',minimum=1);integer(self.p_free,'p_free')
        integer(self.df,'df',minimum=min(0,self.N-self.p_free))
        require(self.N==len(self.time)==len(self.residuals)==len(self.raw_indices) and self.df==self.N-self.p_free,'Noise full-array N/p/df conflict')
        for k in ('raw_SSE','raw_RMSE'):object.__setattr__(self,k,number(getattr(self,k),k,nonnegative=True))
        require(isclose(sum(x*x for x in self.residuals),self.raw_SSE,rel_tol=1e-12,abs_tol=1e-14),'Noise SSE/residual conflict')
        require(isclose(self.raw_RMSE**2*self.N,self.raw_SSE,rel_tol=1e-12,abs_tol=1e-14),'Noise RMSE conflict')
        require((self.RSE is None)==(self.df<=0),'RSE unavailable exactly when N<=p')
        if self.RSE is not None:
            number(self.RSE,'RSE',nonnegative=True)
            require(isclose(self.RSE**2*self.df,self.raw_SSE,rel_tol=1e-12,abs_tol=1e-14),'Noise RSE conflict')
        if self.sigma_noise is not None:number(self.sigma_noise,'sigma_noise',positive=True)
        if self.noise_unit_factor is not None:number(self.noise_unit_factor,'noise_unit_factor',positive=True)
        require((self.sigma_noise is None)==(self.noise_unit_factor is None),'Noise sigma/unit conversion conflict')
        if self.sigma_noise is not None:
            require(self.reference is not None,'Sigma requires its noise reference')
            from pyecr.units import factor_to_reference
            expected=factor_to_reference(self.reference.signal_unit)/factor_to_reference(self.unit)
            require(isclose(expected,self.noise_unit_factor,rel_tol=1e-12) and isclose(self.sigma_noise,self.reference.sigma*expected,rel_tol=1e-12,abs_tol=1e-14),'Noise unit/SD identity conflict')
        if self.rank is not None:integer(self.rank,'rank');require(self.rank<=self.p_free,'Invalid noise Jacobian rank')
        for k in ('lag1','drift_span_over_sd','half_sd_ratio'):
            if getattr(self,k) is not None:number(getattr(self,k),k)
        require(self.lag1 is None or -1.000000000001<=self.lag1<=1.000000000001,'Invalid lag1')
        require(self.status in {'not_set','not_comparable','conditional','stale'},'Unknown noise comparison status')
        require((self.RSE_over_sigma is not None)==(self.status=='conditional'),'Ratio/status conflict')
        require(self.status=='conditional' or bool(self.reasons),'Unavailable comparison needs reasons')
        if self.status=='not_set':require(self.reference is None and self.sigma_noise is None,'Not-set cannot invent sigma')
        if self.status=='conditional':
            number(self.RSE_over_sigma,'RSE_over_sigma',nonnegative=True)
            require(self.reference is not None and self.RSE is not None and self.sigma_noise is not None and not self.reasons,'Conditional ratio lacks evidence')
            require(not self.reference.issues,'Conditional ratio conflicts with noise-reference issues')
            require(self.loss=='linear' and self.weighting=='known_unit_weights' and self.rank==self.p_free,'Conditional ratio requires linear full-rank unit-weight model')
            require(self.reference.assumptions=='iid_stationary_constant_sd','Conditional ratio requires explicit assumptions')
            require((self.reference.source.dataset_sha256,self.reference.source.recipe_sha256)==(self.source.dataset_sha256,self.source.recipe_sha256),'Noise/Fit source conflict')
            require(self.reference.response_profile==self.response_profile,'Noise response profile conflict')
            require(isclose(self.RSE_over_sigma,self.RSE/self.sigma_noise,rel_tol=1e-12,abs_tol=1e-14),'Noise ratio conflict')
