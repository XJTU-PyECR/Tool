"""Opt-in signed-tikhonov/2: the SAME fixed linear objective as version one.

One gesdd factorization supplies both the minimum-norm solution and tr(H).
The projected variant replaces Aw by its economic QR R, retains the ORIGINAL
cutoff and N, and evaluates residuals on every observation. No Gram inverse,
N by N right-hand side, endpoint prior, spectrum clipping or publication.
"""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy import linalg

from pyecr.contracts import ResultState, SignedDctResult, SignedDctSolveDiagnostics
from pyecr.contracts.common import ContractError, require, integer
from pyecr.dct.execution import check_cancel, numeric_phase

ENGINE_VERSION = 'signed-linear-workspace/1'
DEFAULT_CACHE_BYTES = 128*1024**2


def _freeze(a):
    a = np.asarray(a, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(a.shape)


def _key(*parts):
    h = sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            data = np.ascontiguousarray(part, dtype='<f8').tobytes()
            header = str(part.shape).encode()
        else:
            data = json.dumps(part, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
            header = b'json'
        h.update(len(header).to_bytes(8, 'little')); h.update(header)
        h.update(len(data).to_bytes(8, 'little')); h.update(data)
    return h.hexdigest()


def workspace_key(source, config):
    g = config.grid
    return _key(ENGINE_VERSION, np.asarray(source.time_s), source.raw_indices, source.raw_count, source.t0_s,
        [g.tau_min_s, g.tau_max_s, g.n_tau], config.density_measure, config.kernel, config.regularizer,
        config.baseline_mode, config.solver, config.algorithm_version,
        np.asarray(config.observation_weights or np.ones(len(source.signal))), 'original-max-B-shape-times-float64-eps')


@dataclass(frozen=True, slots=True)
class LinearWorkspace:
    key: str
    tau: np.ndarray
    x: np.ndarray
    q: np.ndarray
    time: np.ndarray
    root_w: np.ndarray
    l: np.ndarray
    data_matrix: np.ndarray
    projection: np.ndarray | None
    h: float
    original_shape: tuple
    cutoff: float

    @property
    def nbytes(self):
        return sum(v.nbytes for v in (self.tau,self.x,self.q,self.time,self.root_w,self.l,self.data_matrix,self.projection) if v is not None)


@dataclass(frozen=True, slots=True)
class LinearFactor:
    key: str
    data_u: np.ndarray
    retained_s: np.ndarray
    vt: np.ndarray
    singular_values: np.ndarray
    rank: int
    df: float
    shape: tuple

    @property
    def nbytes(self):
        return sum(v.nbytes for v in (self.data_u,self.retained_s,self.vt,self.singular_values))


class LinearCache:
    """Request-owned bounded LRU. It stores NO spectrum, score or Y projection.

    Static matrices/factors may safely serve different Y/terminals: RHS and
    restored results are always recomputed from the complete frozen source.
    No global cache; no mutable array can be changed through a cached handle.
    This object is for one worker, not shared between concurrent workers.
    """
    def __init__(self, max_bytes=DEFAULT_CACHE_BYTES):
        self.max_bytes = integer(max_bytes, 'cache byte budget')
        self.entries = OrderedDict(); self.bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0

    def get(self, key, build):
        check_cancel()
        if key in self.entries:
            self.hits += 1; self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        value = build()
        check_cancel()
        if value.nbytes > self.max_bytes:
            self.bypasses += 1
            return value
        while self.entries and self.bytes+value.nbytes > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.nbytes; self.evictions += 1
        self.entries[key] = value; self.bytes += value.nbytes
        return value

    def clear(self):
        self.entries.clear(); self.bytes = 0

    def telemetry(self):
        return dict(version=ENGINE_VERSION,max_bytes=self.max_bytes,bytes=self.bytes,entries=len(self.entries),
            hits=self.hits,misses=self.misses,evictions=self.evictions,bypasses=self.bypasses)


def prepare_workspace(source, config, cache):
    key = workspace_key(source, config)
    def build():
        from pyecr.dct.signed import _operators
        with numeric_phase('signed_linear_operators'):
            tau,x,q,a,l,h = _operators(source.time_s, config)
            root = np.sqrt(np.asarray(config.observation_weights or np.ones(len(source.signal))))
            a *= root[:, None]
            require(np.all(np.isfinite(a)), 'nonfinite weighted kernel')
        n,m = a.shape
        projection = None
        if config.solver == 'projected_svd' and n >= m:
            with numeric_phase('signed_linear_qr'):
                projection, a = linalg.qr(a, mode='economic', check_finite=False)
        original_shape = (n+m-2,m)
        return LinearWorkspace(key,*(_freeze(v) for v in (tau,x,q,np.asarray(source.time_s),root,l,a)),
            None if projection is None else _freeze(projection), h,original_shape,max(original_shape)*np.finfo(float).eps)
    return cache.get(('workspace',key),build)


def factor_workspace(workspace, lam, cache):
    key = _key(workspace.key,float(lam),'gesdd',workspace.cutoff)
    def build():
        data = workspace.data_matrix
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            b = np.vstack((data,np.sqrt(lam)*workspace.l))
        require(np.all(np.isfinite(b)), 'nonfinite augmented linear system')
        with numeric_phase('signed_linear_svd'):
            u,s,vt = linalg.svd(b,full_matrices=False,lapack_driver='gesdd',check_finite=False)
        retained = s > s[0]*workspace.cutoff
        rank = int(np.count_nonzero(retained))
        require(rank>0, 'numerically zero augmented rank')
        data_u = u[:len(data),retained]
        df = float(np.sum(data_u*data_u))
        n = len(workspace.time)
        require(np.isfinite(df) and -64*np.finfo(float).eps*n <= df <= n*(1+64*np.finfo(float).eps),
                'nonfinite or invalid linear trace')
        # The trace can overshoot N by roundoff in saturated, unregularized cases.
        df = min(float(n),max(0.,df))
        return LinearFactor(key,_freeze(data_u),_freeze(s[retained]),_freeze(vt[retained]),_freeze(s),rank,df,b.shape)
    return cache.get(('factor',key),build)


def reconstruct(workspace, chi, block_rows=4096):
    out = np.empty(len(workspace.time))
    for start in range(0,len(out),block_rows):
        check_cancel(); stop = min(len(out),start+block_rows)
        with np.errstate(over='ignore',under='ignore'):
            kernel = np.exp(-workspace.time[start:stop,None]/workspace.tau[None,:])*workspace.q
        out[start:stop] = kernel@chi
    return out


def fit_reused(source, config, run_id, cache=None):
    cache = cache if cache is not None else LinearCache()
    try:
        workspace = prepare_workspace(source,config,cache)
        factor = factor_workspace(workspace,config.grid.regularization_lambda,cache)
        # Never cache this projection: it is bound to all Y/normalization bytes.
        with numeric_phase('signed_linear_rhs'):
            yw = workspace.root_w*np.asarray(source.signal)
            rhs = yw if workspace.projection is None else workspace.projection.T@yw
            orthogonal = 0. if workspace.projection is None else float(np.sum((yw-workspace.projection@rhs)**2))
            chi = factor.vt.T@((factor.data_u.T@rhs)/factor.retained_s)
        with numeric_phase('signed_linear_reconstruction'):
            prediction = reconstruct(workspace,chi)
            residual = np.asarray(source.signal)-prediction
            sse = float(np.sum((workspace.root_w*residual)**2))
            penalty_vector = (np.sqrt(config.grid.regularization_lambda)*workspace.l)@chi
            penalty = float(penalty_vector@penalty_vector)
            # Independent full-observation SSE remains authoritative; QR's
            # orthogonal complement is also recorded for projection audits.
            projected_sse = float(np.sum((rhs-workspace.data_matrix@chi)**2))+orthogonal
            require(np.isclose(projected_sse,sse,rtol=2e-8,atol=2e-12*max(1.,float(yw@yw))),
                    'QR projection lost full-observation SSE')
        s = factor.singular_values
        condition = None
        with np.errstate(over='ignore',divide='ignore'):
            ratio = float(s[0]/s[-1])
        if np.isfinite(ratio): condition = ratio
        diagnostics = SignedDctSolveDiagnostics(schema_version=2,
            solver_status='rank_deficient' if factor.rank<len(workspace.tau) else 'converged',
            lapack_driver='gesdd',augmented_shape=workspace.original_shape,effective_rank=factor.rank,svd_rank=factor.rank,
            relative_cutoff=workspace.cutoff,singular_values=tuple(s),condition_number=condition,
            effective_condition_number=float(s[0]/s[factor.rank-1]),grid_step_decades=workspace.h,
            regularizer_rows=len(workspace.l),weighted_sse=sse,regularization_penalty=penalty,objective=sse+penalty,
            effective_df=factor.df,factorization_shape=factor.shape,orthogonal_weighted_sse=orthogonal)
        mass = chi*workspace.q
        require(all(np.all(np.isfinite(v)) for v in (chi,mass,prediction,residual)), 'nonfinite linear solution')
        norm = source.normalization
        require(all(np.all(np.isfinite(v)) for v in (norm.reference_value+norm.scale_value*prediction,
            norm.scale_value*residual,norm.scale_value*chi,norm.scale_value*mass)), 'nonfinite physical restoration')
    except (ContractError,ValueError,FloatingPointError,OverflowError,linalg.LinAlgError) as exc:
        return SignedDctResult(schema_version=2,source=source,config=config,
            state=ResultState(status='error',blocking_reasons=(f'Signed reused solve failed: {exc}',)),run_id=run_id)
    warnings = list(source.state.warnings)+list(norm.state.warnings)
    if factor.rank<len(workspace.tau):
        warnings.append('SIGNED_DCT_RANK_DEFICIENT: numerical truncation does not establish a unique spectrum')
    if condition is None or condition > 1/np.sqrt(np.finfo(float).eps):
        warnings.append('SIGNED_DCT_ILL_CONDITIONED: augmented condition exceeds 1/sqrt(float64 eps); coefficients may be sensitive')
    check_cancel()
    return SignedDctResult(schema_version=2,source=source,config=config,run_id=run_id,
        state=ResultState(status='warning' if warnings else 'completed',warnings=tuple(dict.fromkeys(warnings))),
        tau_s=tuple(workspace.tau),log10_tau=tuple(workspace.x),chi=tuple(chi),quadrature_weights=tuple(workspace.q),
        quadrature_mass=tuple(mass),reconstruction=tuple(prediction),residuals=tuple(residual),
        selected_lambda=config.grid.regularization_lambda,solve_diagnostics=diagnostics)
