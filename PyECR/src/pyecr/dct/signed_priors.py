"""Explicit B1 and separately named 2012/engineering UP/ZC experiments.

Public input/source gates live in fit_signed_dct. The returned main spectrum
exists only when the outer iteration and both declared starts pass. Failed
last iterates remain diagnostic attempts, never a silently published fallback.
"""
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4
import numpy as np
from scipy import linalg
from scipy.ndimage import maximum_filter1d

from pyecr.contracts import (SignedDctResult,SignedDctSolveDiagnostics,SignedDctPriorEvidence,
    SignedPriorIteration,ResultState)
from pyecr.contracts.common import ContractError,require
from pyecr.dct.execution import check_cancel,numeric_phase
from pyecr.dct.signed_linear import LinearCache,prepare_workspace
from pyecr.dct.signed_prior_kernel import bind_problem,b1_blocks,fixed_subproblem,edge_profile


def base_config(config):
    return replace(config,schema_version=2,algorithm_version='signed-tikhonov/2',solver='projected_svd',
        regularizer='sqrt_dlog10_tau_second_derivative_v1',prior_spec=None)


def _maximum(values,radius):
    return maximum_filter1d(values,2*radius+1,mode='constant',cval=0.) if radius else values.copy()


def adaptive_setup(workspace,spec):
    m=len(workspace.x);h=workspace.h;identity=np.eye(m)
    plus=-identity+np.eye(m,k=1);minus=identity-np.eye(m,k=-1)
    central=(plus+minus)/2;second=plus-minus
    if spec.family=='up2012_mass_reference':
        bases=[('curvature',second*workspace.q)]
        if spec.zc_enabled:bases.append(('zc',central*workspace.q))
        if spec.edge_beta:bases.append(('edge',np.sqrt(spec.edge_beta)*np.diag(workspace.q)))
        def weights(chi):
            delta=np.log(10.)*h;mass=workspace.q*chi
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                p=central@mass/delta**2;c=second@mass/delta**3
                normalized=1/np.sqrt(spec.up_floor+spec.up_slope_ratio*_maximum(p*p,spec.reference_neighbour_nodes)
                    +spec.up_curvature_ratio*_maximum(c*c,spec.reference_neighbour_nodes))
                normalized=np.clip(normalized,spec.up_weight_min,spec.up_weight_max)
                curvature=delta**(-2.5)*normalized;result={'curvature':curvature}
                if spec.zc_enabled:
                    sign=np.sign(mass)
                    crossings=_maximum(((minus@sign)/2)**2+((plus@sign)/2)**2,spec.reference_zc_neighbour_nodes)
                    zc=delta**(-1.5)*spec.zc_strength*crossings/np.sqrt(spec.zc_floor+spec.zc_slope_ratio*p*p)
                    result['zc']=np.minimum(zc,spec.zc_cap_relative_up*curvature)
                if spec.edge_beta:
                    mask=edge_profile(workspace.x,spec.buffer_decades,'box')
                    result['edge']=mask*delta**(-.5)/np.sqrt(spec.edge_floor+spec.edge_amplitude_ratio*(mass/delta)**2)
            return result
    else:
        # Interior curvature, true one-sided density slope at endpoints. No
        # zero-extension of density is smuggled into the engineering variant.
        derivative=central/h
        derivative[0]=0;derivative[0,:2]=[-1/h,1/h]
        derivative[-1]=0;derivative[-1,-2:]=[-1/h,1/h]
        radius=int(np.ceil(spec.neighbourhood_decades/h))
        bases=[('curvature',workspace.l)]
        if spec.zc_enabled:bases.append(('zc',np.sqrt(workspace.q)[:,None]*derivative/spec.length_scale_decades))
        active=None
        if spec.edge_beta:
            mask=edge_profile(workspace.x,spec.buffer_decades)
            active=np.flatnonzero(mask>0);edge=np.zeros((len(active),m))
            edge[np.arange(len(active)),active]=np.sqrt(spec.edge_beta*workspace.q[active]*mask[active])
            bases.append(('edge',edge))
        def weights(chi):
            ell=spec.length_scale_decades
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                p=derivative@chi;c=np.r_[0.,workspace.l@chi/np.sqrt(h),0.]
                c[0]=c[1];c[-1]=c[-2]
                raw=1/np.sqrt(spec.up_floor+spec.up_slope_ratio*_maximum((ell*p)**2,radius)
                    +spec.up_curvature_ratio*_maximum((ell*ell*c)**2,radius))
                curvature=np.clip(raw,spec.up_weight_min,spec.up_weight_max)
                result={'curvature':curvature[1:-1]}
                if spec.zc_enabled:
                    sign=np.sign(chi)
                    backward=sign-np.r_[sign[0],sign[:-1]];forward=np.r_[sign[1:],sign[-1]]-sign
                    crossings=_maximum((backward**2+forward**2)/4,radius)
                    zc=spec.zc_strength*crossings/np.sqrt(spec.zc_floor+spec.zc_slope_ratio*(ell*p)**2)
                    result['zc']=np.minimum(zc,spec.zc_cap_relative_up*curvature)
                if spec.edge_beta:
                    result['edge']=(1/np.sqrt(spec.edge_floor+spec.edge_amplitude_ratio*chi[active]**2)
                        if spec.adaptive_edge else np.ones(len(active)))
            return result
    return tuple(bases),weights


def _blocks(bases,weights):
    require(all(np.all(np.isfinite(w)) and np.all(w>=0) for w in weights.values()),'invalid adaptive weight')
    return tuple((name,weights[name][:,None]*operator) for name,operator in bases)


def _weight_vector(weights):return np.concatenate([weights[k] for k in sorted(weights)])


def _regularizer_hash(blocks):
    h=sha256()
    for name,op in blocks:h.update(name.encode());h.update(np.ascontiguousarray(op,dtype='<f8').tobytes())
    return h.hexdigest()


def _one_initial(problem,lam,spec,initial,label,bases,weight_function):
    chi=np.array(initial,copy=True);prediction=problem.workspace.data_matrix@chi
    weights=weight_function(chi);trace=[];history=[];changes=[];stable=0;last=None;blocks=None
    status='FAIL_MAX_ITER';reason='maximum outer iteration budget exhausted'
    for iteration in range(1,spec.max_iterations+1):
        check_cancel();blocks=_blocks(bases,weights)
        previous_sse=float(np.sum((problem.projected_y-prediction)**2))+problem.orthogonal_sse
        before=previous_sse+sum(lam*float(np.sum((r@chi)**2)) for _,r in blocks)
        try:
            result=fixed_subproblem(problem,lam,blocks)
            last=result
            proposed=weight_function(result.chi)
        except (ContractError,ValueError,FloatingPointError,OverflowError,linalg.LinAlgError) as exc:
            return last,tuple(trace),'FAIL_NUMERIC',f'iteration {iteration}: {exc}',blocks
        old_vector=_weight_vector(weights);new_vector=_weight_vector(proposed)
        dc=float(np.linalg.norm(result.chi-chi)/max(np.linalg.norm(chi),np.linalg.norm(result.chi),1e-12))
        dp=float(np.linalg.norm(problem.workspace.data_matrix@(result.chi-chi))/max(np.linalg.norm(problem.yw),1e-12))
        dw=float(np.linalg.norm(new_vector-old_vector)/max(np.linalg.norm(old_vector),1e-12))
        trace.append(SignedPriorIteration(initial_label=label,iteration=iteration,fixed_objective_before=before,
            fixed_objective_after=result.objective,weighted_sse=result.weighted_sse,penalties=result.penalties,
            chi_change=dc,prediction_change=dp,weight_change=dw,optimality=result.optimality,damping=spec.damping,
            rank=result.rank,cutoff=result.cutoff,weight_ranges=tuple((name,float(np.min(v)),float(np.max(v))) for name,v in weights.items()),
            weights_sha256=_regularizer_hash(blocks),seconds=result.seconds))
        last=result
        if result.optimality>spec.optimality_tolerance or result.objective>before+1e-10*max(1.,before):
            status='FAIL_NUMERIC';reason='fixed subproblem failed declared optimality/decrease check';break
        good=dc<=spec.chi_tolerance and dp<=spec.prediction_tolerance and dw<=spec.weight_tolerance
        stable=stable+1 if good else 0
        if stable>=2:status='CONVERGED';reason='';break
        if len(history)>=2:
            cycle=float(np.linalg.norm(result.chi-history[-2])/max(np.linalg.norm(result.chi),1e-12))
            if cycle<spec.cycle_tolerance and dc>10*spec.chi_tolerance:
                status='FAIL_CYCLE';reason='two-cycle detected in density iterates';break
        history.append(result.chi.copy());history=history[-3:]
        changes.append(max(dc,dw,dp));changes=changes[-spec.stagnation_patience:]
        if len(changes)==spec.stagnation_patience and min(changes)>max(spec.chi_tolerance,spec.weight_tolerance) and max(changes)/min(changes)<1.001:
            status='FAIL_STALLED';reason='outer changes stagnated above declared tolerances';break
        chi=np.asarray(result.chi);prediction=problem.workspace.data_matrix@chi
        weights={name:(1-spec.damping)*weights[name]+spec.damping*proposed[name] for name in weights}
    return last,tuple(trace),status,reason,blocks


def fit_prior(source,config,run_id=None,cache=None):
    run_id=run_id or uuid4().hex;spec=config.prior_spec;cache=cache if cache is not None else LinearCache()
    trace=();statuses=();attempt=None;last_blocks=None;delta=None;dp=None
    try:
        w=prepare_workspace(source,base_config(config),cache);problem=bind_problem(w,source);lam=config.grid.regularization_lambda
        if not spec.adaptive:
            last_blocks=b1_blocks(w,spec.alpha1,spec.alpha0,spec.edge_beta,spec.buffer_decades)
            solution=fixed_subproblem(problem,lam,last_blocks,hard_zero=spec.hard_zero_control)
            status='LINEAR_CONVERGED';selected='linear';df_semantics='fixed_linear';warnings=()
        else:
            bases,weight_function=adaptive_setup(w,spec)
            initial=fixed_subproblem(problem,lam,b1_blocks(w,spec.alpha1,spec.alpha0,spec.initial_edge_beta,spec.buffer_decades))
            first,t1,s1,r1,b1=_one_initial(problem,lam,spec,initial.chi,'B1',bases,weight_function)
            second,t2,s2,r2,b2=_one_initial(problem,lam,spec,np.zeros(len(w.x)),'zero',bases,weight_function)
            trace=t1+t2;statuses=(('B1',s1),('zero',s2));attempt=first or second;last_blocks=b1 or b2
            df_semantics='conditional_frozen_weights';warnings=('CONDITIONAL_FROZEN_WEIGHT_DF_NOT_NONLINEAR_GCV',)
            if first is not None and second is not None:
                delta=float(np.sum(w.q*np.abs(first.chi-second.chi))/max(np.sum(w.q*np.abs(first.chi)),np.sum(w.q*np.abs(second.chi)),1e-12))
                dp=float(np.linalg.norm(first.prediction-second.prediction)/max(np.linalg.norm(problem.y),1e-12))
            if s1!='CONVERGED' or s2!='CONVERGED':
                status=s1 if s1!='CONVERGED' else s2
                reasons=tuple(f'{label}: {message}' for label,message in (('B1',r1),('zero',r2)) if message)
            elif delta>spec.initial_density_tolerance or dp>spec.initial_prediction_tolerance:
                status='FAIL_INITIAL_DEPENDENCE';reasons=('Declared initializations converge to materially different spectra/reconstructions',)
            else:status='CONVERGED';reasons=()
            if status.startswith('FAIL'):
                evidence=SignedDctPriorEvidence(family=spec.family,status=status,df_semantics=df_semantics,
                    iteration_trace=trace,initial_statuses=statuses,initial_density_difference=delta,initial_prediction_difference=dp,
                    failure_reasons=reasons,warning_flags=warnings,attempted_chi=tuple(attempt.chi) if attempt else (),
                    attempted_reconstruction=tuple(attempt.prediction) if attempt else ())
                return SignedDctResult(schema_version=3,source=source,config=config,run_id=run_id,
                    state=ResultState(status='error',blocking_reasons=reasons),prior_evidence=evidence)
            solution=first;selected='B1'
        singular=solution.singular_values
        with np.errstate(over='ignore',divide='ignore'):condition=float(singular[0]/singular[-1])
        if not np.isfinite(condition):condition=None
        diagnostics=SignedDctSolveDiagnostics(schema_version=3,solver_status='converged' if solution.rank==len(solution.free_indices) else 'rank_deficient',
            lapack_driver='gesdd',augmented_shape=solution.augmented_shape,effective_rank=solution.rank,svd_rank=solution.rank,
            relative_cutoff=solution.cutoff,singular_values=tuple(singular),condition_number=condition,
            effective_condition_number=float(singular[0]/singular[solution.rank-1]),grid_step_decades=w.h,
            regularizer_rows=solution.augmented_shape[0]-len(w.time),weighted_sse=solution.weighted_sse,
            regularization_penalty=sum(v for _,v in solution.penalties),objective=solution.objective,
            effective_df=solution.df_frozen,factorization_shape=solution.factor_shape,orthogonal_weighted_sse=problem.orthogonal_sse,df_semantics=df_semantics)
        evidence=SignedDctPriorEvidence(family=spec.family,status=status,df_semantics=df_semantics,penalties=solution.penalties,
            free_indices=solution.free_indices,iteration_trace=trace,initial_statuses=statuses,initial_density_difference=delta,
            initial_prediction_difference=dp,selected_initial=selected,regularizer_sha256=_regularizer_hash(last_blocks),
            warning_flags=warnings+(('HARD_ZERO_BOUNDARY_SENSITIVITY_CONTROL',) if spec.hard_zero_control else ()))
        warnings=tuple(dict.fromkeys((*source.state.warnings,*source.normalization.state.warnings,*evidence.warning_flags,
            'EXPLICIT_SIGNED_PRIOR_CONDITIONAL_ON_DECLARED_REGULARIZER',
            *(('SIGNED_DCT_RANK_DEFICIENT',) if solution.rank<len(solution.free_indices) else ()),
            *(('SIGNED_DCT_ILL_CONDITIONED',) if condition is None or condition>1/np.sqrt(np.finfo(float).eps) else ()))))
        check_cancel()
        return SignedDctResult(schema_version=3,source=source,config=config,run_id=run_id,state=ResultState(status='warning',warnings=warnings),
            tau_s=tuple(w.tau),log10_tau=tuple(w.x),chi=tuple(solution.chi),quadrature_weights=tuple(w.q),
            quadrature_mass=tuple(w.q*solution.chi),reconstruction=tuple(solution.prediction),residuals=tuple(solution.residuals),
            selected_lambda=float(lam),solve_diagnostics=diagnostics,prior_evidence=evidence)
    except (ContractError,ValueError,FloatingPointError,OverflowError,linalg.LinAlgError) as exc:
        reason=f'Prior numerical failure: {exc}'
        evidence=SignedDctPriorEvidence(family=spec.family,status='FAIL_NUMERIC',df_semantics='conditional_frozen_weights' if spec.adaptive else 'fixed_linear',
            failure_reasons=(reason,),iteration_trace=trace,initial_statuses=statuses)
        return SignedDctResult(schema_version=3,source=source,config=config,run_id=run_id,
            state=ResultState(status='error',blocking_reasons=(reason,)),prior_evidence=evidence)
