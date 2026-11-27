"""Explicit signed convex experiment; never a default or current publisher.

J=SSE+lambda*||L2 chi||^2+alpha*sum(q*abs(chi))+beta*sum(q*H*chi^2).
chi=B a, where B is identity or a nonnegative cubic B-spline partition of
unity. The L1 term is the EXACT nodal absolute mass, not coefficient L1.
ADMM splits z=Q B a. A single augmented SVD is reused within each solve;
no normal-equation Gram matrix, inverse, nonlinear GCV or positivity.
"""
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from pyecr.contracts.common import require
from pyecr.dct.execution import check_cancel, numeric_phase
from pyecr.dct.signed_linear import LinearCache, prepare_workspace, _freeze, _key, reconstruct
from pyecr.dct.signed_prior_kernel import edge_profile

VERSION='signed-convex-mass/1'


@dataclass(frozen=True,slots=True)
class ConvexSpec:
    version:str=VERSION
    knot_step_decades:float=0.
    mass_ratio:float=.25
    edge_beta:float=0.
    edge_width_decades:float=.5
    ou_correlation_seconds:float=0.
    rho:float=1.
    max_iterations:int=6000
    abs_tolerance:float=1e-8
    rel_tolerance:float=1e-5
    kkt_tolerance:float=2e-5

    def __post_init__(self):
        require(self.version==VERSION,'unsupported convex version')
        for name in ('knot_step_decades','mass_ratio','edge_beta','ou_correlation_seconds'):
            v=getattr(self,name);require(isfinite(v) and v>=0,'invalid '+name)
        for name in ('rho','edge_width_decades','abs_tolerance','rel_tolerance','kkt_tolerance'):
            require(isfinite(getattr(self,name)) and getattr(self,name)>0,'invalid '+name)
        require(type(self.max_iterations) is int and 1<=self.max_iterations<=100000,'invalid iteration budget')
        require(self.knot_step_decades==0 or self.knot_step_decades>=.025,'knot spacing too small')

    def to_dict(self):return asdict(self)


def density_basis(x,step):
    if not step:return np.eye(len(x))
    count=max(1,int(np.ceil((x[-1]-x[0])/step)))
    knots=np.r_[np.repeat(x[0],4),np.linspace(x[0],x[-1],count+1)[1:-1],np.repeat(x[-1],4)]
    b=BSpline.design_matrix(x,knots,3,extrapolate=False).toarray()
    require(np.allclose(b.sum(axis=1),1.,rtol=0,atol=2e-14),'basis partition failed')
    return b


def whiten(values,time,correlation):
    """Exact stationary OU innovations on actual times; O(N), no N*N matrix."""
    values=np.asarray(values)
    if not correlation:return values.copy()
    rho=np.exp(-np.diff(time)/correlation)
    require(np.all(rho<1),'OU whitening needs strictly increasing time')
    denominator=np.sqrt(-np.expm1(-2*np.diff(time)/correlation))
    out=values.copy()
    if out.ndim==1:out[1:]=(values[1:]-rho*values[:-1])/denominator
    else:out[1:]=(values[1:]-rho[:,None]*values[:-1])/denominator[:,None]
    return out


@dataclass(frozen=True,slots=True)
class ConvexWorkspace:
    key:str
    base:object
    basis:np.ndarray
    data:np.ndarray
    projection:np.ndarray|None
    mass:np.ndarray
    curvature:np.ndarray
    edge:np.ndarray
    singular_values:np.ndarray

    @property
    def nbytes(self):return sum(a.nbytes for a in (self.basis,self.data,self.projection,self.mass,self.curvature,self.edge,self.singular_values) if a is not None)


def prepare_convex(source,config,spec,cache):
    require(config.solver=='reused_svd' and config.grid.lambda_mode=='manual','explicit unprojected v2 static workspace required')
    source.require_usable()
    require(source.normalization is not None and source.component_key is None,'Whole terminal-referenced source required')
    base=prepare_workspace(source,config,cache)
    key=_key(base.key,spec.to_dict(),'convex-static/1')
    def build():
        b=density_basis(base.x,spec.knot_step_decades)
        a=whiten(base.data_matrix@b,base.time,spec.ou_correlation_seconds)
        s=linalg.svdvals(a,check_finite=False)
        projection=None
        if len(a)>=a.shape[1]:projection,a=linalg.qr(a,mode='economic',check_finite=False)
        h=edge_profile(base.x,spec.edge_width_decades)
        return ConvexWorkspace(key,base,*(_freeze(v) if v is not None else None for v in
            (b,a,projection,base.q[:,None]*b,base.l@b,np.sqrt(base.q*h)[:,None]*b,s)))
    return cache.get(('convex_static',key),build)


@dataclass(frozen=True,slots=True)
class ConvexFactor:
    left:np.ndarray
    vt:np.ndarray
    singular_values:np.ndarray
    rank:int
    cutoff:float
    @property
    def nbytes(self):return self.left.nbytes+self.vt.nbytes+self.singular_values.nbytes


def solve_convex(source,config,spec,*,cache=None):
    """Solve one frozen lambda. Failure keeps diagnostics but never a main spectrum."""
    start=perf_counter();check_cancel();cache=cache if cache is not None else LinearCache()
    w=prepare_convex(source,config,spec,cache);lam=float(config.grid.regularization_lambda)
    alpha=spec.mass_ratio*np.sqrt(lam);beta=spec.edge_beta*lam
    yw=whiten(w.base.root_w*np.asarray(source.signal),w.base.time,spec.ou_correlation_seconds)
    rhs=yw if w.projection is None else w.projection.T@yw
    orth=0. if w.projection is None else float(np.sum((yw-w.projection@rhs)**2))
    d=len(w.data);rows=[w.data,np.sqrt(lam)*w.curvature,np.sqrt(beta)*w.edge]
    split=alpha>0
    if split:rows.append(np.sqrt(spec.rho/2)*w.mass)
    key=_key(w.key,lam,alpha,beta,spec.rho,'convex-factor/1')
    def factor():
        matrix=np.vstack(rows)
        with numeric_phase('signed_convex_augmented_svd'):
            u,s,vt=linalg.svd(matrix,full_matrices=False,lapack_driver='gesdd',check_finite=False)
        cutoff=max(len(yw)+len(matrix)-d,matrix.shape[1])*np.finfo(float).eps
        keep=s>s[0]*cutoff
        return ConvexFactor(_freeze(u[:,keep]/s[keep]),_freeze(vt[keep]),_freeze(s),int(keep.sum()),cutoff)
    f=cache.get(('convex_factor',key),factor)
    constant=f.left[:d].T@rhs
    z=np.zeros(len(w.base.q));u=z.copy();trace=[];ok=False;failure='FAIL_MAX_ITER'
    kkt=primal=dual=0.;a=np.zeros(w.basis.shape[1])
    for iteration in range(1,spec.max_iterations+1 if split else 2):
        check_cancel()
        coeff=constant
        if split:coeff=coeff+np.sqrt(spec.rho/2)*(f.left[-len(z):].T@(z-u))
        a=f.vt.T@coeff;mass=w.mass@a
        if split:
            old=z.copy();v=mass+u;z=np.sign(v)*np.maximum(np.abs(v)-alpha/spec.rho,0.);u+=mass-z
            primal=float(np.linalg.norm(mass-z));dual=float(spec.rho*np.linalg.norm(w.mass.T@(z-old)))
            epsp=np.sqrt(len(z))*spec.abs_tolerance+spec.rel_tolerance*max(np.linalg.norm(mass),np.linalg.norm(z))
            epsd=np.sqrt(len(a))*spec.abs_tolerance+spec.rel_tolerance*np.linalg.norm(spec.rho*w.mass.T@u)
        else:epsp=epsd=0.
        if iteration%25==0 or iteration==1 or not split or (primal<=epsp and dual<=epsd) or iteration==spec.max_iterations:
            residual=w.data@a-rhs
            grad=2*w.data.T@residual+2*lam*w.curvature.T@(w.curvature@a)+2*beta*w.edge.T@(w.edge@a)
            dual_gradient=spec.rho*w.mass.T@u if split else np.zeros_like(grad)
            scale=max(1.,np.linalg.norm(2*w.data.T@rhs))
            kkt=float(np.linalg.norm(grad+dual_gradient)/scale)
            objective=float(residual@residual+orth+lam*np.sum((w.curvature@a)**2)+beta*np.sum((w.edge@a)**2)+alpha*np.abs(mass).sum())
            trace.append(dict(iteration=iteration,objective=objective,primal=primal,dual=dual,kkt=kkt))
            if not np.isfinite(objective):failure='FAIL_NONFINITE';break
            if primal<=epsp and dual<=epsd and kkt<=spec.kkt_tolerance:ok=True;break
    chi=w.basis@a;pred=reconstruct(w.base,chi);res=np.asarray(source.signal)-pred
    white_res=whiten(w.base.root_w*res,w.base.time,spec.ou_correlation_seconds)
    sse=float(white_res@white_res);penalty=float(lam*np.sum((w.curvature@a)**2));edge=float(beta*np.sum((w.edge@a)**2))
    require(np.isclose(sse,float(np.sum((w.data@a-rhs)**2))+orth,rtol=2e-8,atol=1e-11*max(1.,float(yw@yw))),'convex QR lost SSE')
    check_cancel()
    result=dict(version=VERSION,source_sha256=source.content_sha256,spec=spec.to_dict(),lambda_value=lam,alpha_mass=alpha,
        status='CONVERGED' if ok else failure,fallback=None,iterations=iteration,trace=trace,seconds=perf_counter()-start,
        weighted_sse=sse,curvature_penalty=penalty,edge_penalty=edge,absolute_mass_penalty=float(alpha*np.abs(mass).sum()),
        objective=sse+penalty+edge+float(alpha*np.abs(mass).sum()),orthogonal_sse=orth,kkt=kkt,primal=primal,dual=dual,
        basis_dimension=w.basis.shape[1],original_n=len(yw),rank=f.rank,cutoff=f.cutoff,
        information_singular_values=w.singular_values.tolist(),information_rank_relative_1e3=int(np.sum(w.singular_values>w.singular_values[0]*1e-3)),
        information_note='relative singular-value count is a diagnostic, not a confidence level or hard cutoff',
        tau_s=w.base.tau.tolist(),quadrature_weights=w.base.q.tolist(),
        chi=chi.tolist() if ok else None,quadrature_mass=mass.tolist() if ok else None,
        reconstruction=pred.tolist() if ok else None,residuals=res.tolist() if ok else None,
        diagnostics=None,df=None,gcv=None,df_note='No frozen quadratic df is presented as full L1 GCV')
    if ok:
        positive=float(np.maximum(mass,0).sum());negative=float(np.maximum(-mass,0).sum())
        x=w.base.x;edge_mask=(x<x[0]+.5)|(x>x[-1]-.5)
        result['diagnostics']=dict(positive_mass=positive,negative_mass_magnitude=negative,absolute_mass=positive+negative,
            signed_mass=float(mass.sum()),edge_absolute_mass=float(np.abs(mass[edge_mask]).sum()),
            cancellation_ratio=(positive+negative)/max(abs(float(mass.sum())),1e-15),
            observation_support_counts=[int(np.sum((w.base.time>=.1*t)&(w.base.time<=3*t))) for t in w.base.tau],
            interpretation='conditional on declared terminal, scale, noise and basis; no H/O assignment or spectrum CI')
    return result


def search_convex(source,config,spec,*,noise_sd_physical,noise_evidence,lambdas=None,max_candidates=13):
    """Known-noise discrepancy only. No truth, peak count or nonlinear GCV."""
    require(bool(noise_evidence),'independent noise evidence required')
    sd=np.asarray(noise_sd_physical,dtype=float)
    require(sd.shape==(len(source.signal),) and np.all(np.isfinite(sd)) and np.all(sd>0),'trusted positive physical SD vector required')
    require(type(max_candidates) is int and 3<=max_candidates<=33,'invalid search budget')
    from dataclasses import replace
    cache=LinearCache();base=prepare_workspace(source,config,cache)
    normalized=sd/source.normalization.scale_value*base.root_w
    require(np.allclose(normalized,normalized[0],rtol=1e-6,atol=0),'whitening assumes supplied precision equalizes marginal SD')
    target=float(len(sd)*normalized[0]**2)
    values=tuple(lambdas or np.logspace(-10,-2,9));require(len(values)<=max_candidates and all(np.isfinite(v) and v>0 for v in values),'invalid lambda grid')
    trials=[];start=perf_counter()
    def add(lam):
        check_cancel();r=solve_convex(source,replace(config,grid=replace(config.grid,regularization_lambda=float(lam))),spec,cache=cache)
        r['selection_score']=abs(np.log(max(r['weighted_sse'],np.finfo(float).tiny)/target)) if r['status']=='CONVERGED' else None
        trials.append(r)
    for lam in values:add(lam)
    good=[r for r in trials if r['selection_score'] is not None]
    if good and len(trials)<max_candidates:
        best=min(good,key=lambda r:(r['selection_score'],-r['lambda_value']))['lambda_value']
        ordered=sorted(values);i=ordered.index(best)
        lo=ordered[i-1] if i else best/100;hi=ordered[i+1] if i+1<len(ordered) else best*100
        for v in np.geomspace(lo,hi,7)[1:-1]:
            if len(trials)>=max_candidates:break
            if not any(np.isclose(v,t['lambda_value'],rtol=1e-12,atol=0) for t in trials):add(v)
    good=[r for r in trials if r['selection_score'] is not None]
    selected=min(good,key=lambda r:(r['selection_score'],-r['lambda_value'])) if good else None
    tie=[]
    if selected:
        tie=[r['lambda_value'] for r in good if np.isclose(r['selection_score'],selected['selection_score'],rtol=1e-10,atol=1e-18)]
        selected=next(r for r in good if r['lambda_value']==max(tie))
    return dict(version='signed-convex-search/1',source=source.to_dict(),config=config.to_dict(),spec=spec.to_dict(),
        noise_evidence=noise_evidence,noise_sd_physical=sd.tolist(),strategy='known_noise_discrepancy',target_sse=target,
        trials=trials,selected_lambda=selected['lambda_value'] if selected else None,tie_lambdas=tie,
        selection_boundary=bool(selected and selected['lambda_value'] in (min(r['lambda_value'] for r in trials),max(r['lambda_value'] for r in trials))),
        termination='candidate_budget' if len(trials)==max_candidates else 'frozen_refinement_complete',max_candidates=max_candidates,
        seconds=perf_counter()-start,cache=cache.telemetry(),default_changed=False,experimental=True)
