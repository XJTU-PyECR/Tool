"""Numerical primitives for explicit Signed priors, never a default algorithm.

Uses the stage-one immutable Aw/QR workspace and the same density/quadrature.
Every changed regularizer is factorized afresh. No Gram inverse or positivity.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
import numpy as np
from scipy import linalg
from pyecr.contracts.common import require
from pyecr.dct.execution import check_cancel,numeric_phase
from pyecr.dct.signed_linear import _freeze,reconstruct

_subproblem_counter=ContextVar('signed_prior_subproblem_counter',default=None)


@contextmanager
def count_subproblems():
    counter=[0];token=_subproblem_counter.set(counter)
    try:yield counter
    finally:_subproblem_counter.reset(token)


def derivative_operators(workspace):
    m=len(workspace.tau);h=workspace.h
    first=np.zeros((m-1,m));i=np.arange(m-1)
    first[i,i]=-1/np.sqrt(h);first[i,i+1]=1/np.sqrt(h)
    return workspace.l,first,np.diag(np.sqrt(workspace.q))


def edge_profile(x,width,shape='cosine'):
    require(np.isfinite(width) and width>0,'edge width must be positive decades')
    require(shape in {'cosine','box'},'unknown edge mask')
    distance=np.minimum(x-x[0],x[-1]-x)
    inside=distance<width
    weights=np.zeros(len(x))
    weights[inside]=1. if shape=='box' else .5*(1+np.cos(np.pi*distance[inside]/width))
    return weights


def b1_blocks(workspace,alpha1=0.,alpha0=0.,beta=0.,buffer=.5,shape='cosine'):
    require(all(np.isfinite(v) and v>=0 for v in (alpha1,alpha0,beta)),'negative/nonfinite prior coefficient')
    l2,l1,l0=derivative_operators(workspace)
    blocks=[('curvature',l2)]
    if alpha1:blocks.append(('slope',np.sqrt(alpha1)*l1))
    if alpha0:blocks.append(('amplitude',np.sqrt(alpha0)*l0))
    if beta:
        weights=edge_profile(workspace.x,buffer,shape)
        active=np.flatnonzero(weights>0)
        rows=np.zeros((len(active),len(workspace.x)))
        rows[np.arange(len(active)),active]=np.sqrt(beta*workspace.q[active]*weights[active])
        blocks.append(('edge',rows))
    return tuple(blocks)


@dataclass(frozen=True,slots=True)
class BoundPriorProblem:
    workspace: object
    source_sha256: str
    y: np.ndarray
    yw: np.ndarray
    projected_y: np.ndarray
    orthogonal_sse: float


def bind_problem(workspace,source):
    require(tuple(workspace.time)==source.time_s,'workspace/source time conflict')
    y=np.asarray(source.signal);yw=workspace.root_w*y
    projected=yw if workspace.projection is None else workspace.projection.T@yw
    orthogonal=0. if workspace.projection is None else float(np.sum((yw-workspace.projection@projected)**2))
    return BoundPriorProblem(workspace,source.content_sha256,_freeze(y),_freeze(yw),_freeze(projected),orthogonal)


@dataclass(frozen=True,slots=True)
class PriorSubproblem:
    chi: np.ndarray
    prediction: np.ndarray
    residuals: np.ndarray
    singular_values: np.ndarray
    rank: int
    cutoff: float
    df_frozen: float
    weighted_sse: float
    penalties: tuple
    objective: float
    optimality: float
    augmented_shape: tuple
    factor_shape: tuple
    free_indices: tuple
    seconds: float
    response_matrix: np.ndarray | None = None


def fixed_subproblem(problem,lam,blocks,*,hard_zero=False,resolution=False):
    require(np.isfinite(lam) and lam>=0,'invalid lambda')
    counter=_subproblem_counter.get()
    if counter is not None:counter[0]+=1
    check_cancel();start=perf_counter();w=problem.workspace
    m=len(w.tau);free=np.arange(1,m-1) if hard_zero else np.arange(m)
    regularizer=np.vstack([r for _,r in blocks])
    require(regularizer.shape[1]==m and np.all(np.isfinite(regularizer)),'nonfinite/malformed prior operator')
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        b=np.vstack((w.data_matrix[:,free],np.sqrt(lam)*regularizer[:,free]))
    shape=(len(w.time)+len(regularizer),len(free));cutoff=max(shape)*np.finfo(float).eps
    require(np.all(np.isfinite(b)),'nonfinite augmented prior system')
    with numeric_phase('signed_prior_svd'):
        u,s,vt=linalg.svd(b,full_matrices=False,lapack_driver='gesdd',check_finite=False)
    keep=s>s[0]*cutoff;rank=int(np.count_nonzero(keep));require(rank>0,'zero prior numerical rank')
    data_u=u[:len(w.data_matrix),keep]
    z=vt[keep].T@((data_u.T@problem.projected_y)/s[keep])
    chi=np.zeros(m);chi[free]=z
    with numeric_phase('signed_prior_reconstruction'):
        pred=reconstruct(w,chi);residual=problem.y-pred
        sse=float(np.sum((w.root_w*residual)**2))
        penalties=tuple((name,float(lam*np.sum((r@chi)**2))) for name,r in blocks)
        rhs=np.r_[problem.projected_y,np.zeros(len(regularizer))]
        error=b@z-rhs
        denominator=s[0]*(s[0]*np.linalg.norm(z)+np.linalg.norm(rhs))
        optimality=float(np.linalg.norm(b.T@error)/max(denominator,np.finfo(float).tiny))
        projected_sse=float(np.sum((w.data_matrix@chi-problem.projected_y)**2))+problem.orthogonal_sse
        require(np.isclose(projected_sse,sse,rtol=2e-8,atol=2e-12*max(1.,float(problem.yw@problem.yw))),
                'prior projection lost orthogonal SSE')
    df=float(np.sum(data_u**2));require(df<=len(w.time)*(1+64*np.finfo(float).eps),'invalid conditional trace')
    df=min(float(len(w.time)),df)
    response=None
    if resolution:
        response=np.zeros((m,m))
        response[free,:]=(vt[keep].T/s[keep])@(data_u.T@w.data_matrix)
        response=_freeze(response)
    require(all(np.all(np.isfinite(v)) for v in (chi,pred,residual)),'nonfinite prior solution')
    check_cancel()
    return PriorSubproblem(_freeze(chi),_freeze(pred),_freeze(residual),_freeze(s),rank,cutoff,df,sse,penalties,
        sse+sum(v for _,v in penalties),optimality,shape,b.shape,tuple(int(i) for i in free),perf_counter()-start,response)
