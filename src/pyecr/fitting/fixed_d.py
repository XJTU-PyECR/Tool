"""Opt-in finite-D conditional k inference, separate from legacy staged fitting."""
from pyecr.fitting.reproducibility import seeded_analysis_fit,record_randomness

from dataclasses import replace
import numpy as np
from pyecr.models.ids import ModelID
from pyecr.models.factory import ModelFactory
from pyecr.models.surface_k import SurfaceKKernel
from pyecr.fitting.single import fit_single_ecr,_fit_statistics,_bound_warnings
from pyecr.fitting.dual_surface_k import starts_for_joint
from pyecr.fitting.joint_trf import solve_joint
from pyecr.fitting.base import FitCandidate,FitResult
from pyecr.fitting.solver_strategy import SolverOptions
from pyecr.uncertainty.local import estimate_local_uncertainty
from pyecr.services.fixed_d import validate_fixed_diffusion,VERSION

@seeded_analysis_fit
def fit_fixed_diffusion(dataset,config,*,cancel_check=None,progress_callback=None):
    config.validate()
    sources=validate_fixed_diffusion(config)
    if config.model.boundary_mode!="ideal_step" or config.fit.loss!="linear" or SolverOptions.from_config(config).strategy!="trf":
        raise ValueError("P69B explicit fixed-D entry supports ideal_step/linear/TRF only")
    if config.model_id is ModelID.M1_SINGLE_DK:
        # The existing fitter removes D exactly. Original/default callers never enter here.
        single=dict(config.model_specific.get("single_ecr",{}))
        single.setdefault("normalized_data",dataset.metadata.get("signal_type")=="normalized")
        single.setdefault("n_starts",8)
        internal=replace(config,model_specific={**config.model_specific,"single_ecr":single})
        f=fit_single_ecr(dataset,internal,cancel_check=cancel_check,progress_callback=progress_callback)
        if f.exit_status<=0: raise RuntimeError("Conditional M1 did not converge; no publication")
        return _annotate(dataset,config,f,sources,extra={"underlying_fitter":"unchanged M1 exact fixed-coordinate removal",
            "effective_single_options":single,
            "observation_fixed":{n:(v if single["normalized_data"] else config.model.fixed_parameters.get(n,v)) for n,v in (("amplitude",1.),("baseline",0.)) if n not in f.params}})
    if config.model_id is not ModelID.M3_DUAL_HO_DK: raise ValueError("P69B supports M1/M3 only")
    if dataset.metadata.get("time_unit","s")!="s": raise ValueError("Conditional M3 requires seconds")
    model=ModelFactory().create(config)
    raw=config.model_specific.get("dual_ho",{})
    label_constraint=raw.get("label_constraint","D_H>=D_O")
    if label_constraint not in {"D_H>=D_O","dh_ge_do","none"}: raise ValueError("Unknown H/O label constraint")
    fixed=dict(config.model.fixed_parameters)
    if label_constraint!="none" and fixed["D_H_cm2_s"]<fixed["D_O_cm2_s"]:
        raise ValueError("Fixed D violates selected D_H>=D_O; explicitly choose none if appropriate")
    if {"k_H_cm_s","k_O_cm_s"}&set(fixed): raise ValueError("P69B dual entry fits both k; fixed k Unsupported")
    standard=("k_H_cm_s","k_O_cm_s","A_H","A_O","y0")
    declared=config.model.free_parameters or ("k_H_cm_s","k_O_cm_s","A_H","A_O")
    names=tuple(n for n in standard if n in declared)
    if not {"k_H_cm_s","k_O_cm_s"}.issubset(names):raise ValueError("Both conditional M3 k must be free")
    for n,v in (("A_H",.1),("A_O",.1),("y0",float(dataset.processed_signal[0]))):
        if n not in names:fixed.setdefault(n,v)
    t,y=dataset.processed_time,dataset.processed_signal
    scale=max(float(np.ptp(y)),float(np.max(np.abs(y))),1.)
    defaults={"k_H_cm_s":(1e-10,1.),"k_O_cm_s":(1e-10,1.),"A_H":(0.,10*scale),"A_O":(0.,10*scale),
              "y0":(float(np.min(y)-2*scale),float(np.max(y)+2*scale))}
    bounds={n:config.model.parameter_bounds.get(n,defaults[n]) for n in names}
    initial={"k_H_cm_s":1e-4,"k_O_cm_s":1e-4,**config.model.initial_values}
    options={"term_tol":config.solver.relative_tolerance,"max_terms":config.solver.max_terms}
    def predict(params):
        from pyecr.fitting.iteration_observation import observe_direct_forward
        observe_direct_forward()
        return model.evaluate(t,**params,kernel_options=options).total
    def check():
        if cancel_check:cancel_check()
    count=raw.get("m3_n_starts",20);seed=raw.get("random_seed",20260915)
    if isinstance(count,bool) or not isinstance(count,int) or count<1 or isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError("Conditional starts/seed must be positive/nonnegative integers")
    g=SurfaceKKernel(config.model.geometry).g_cm_inv  # initialization heuristic, not finite-D tau
    starts,profiles=starts_for_joint(t,y,names,fixed,bounds,initial,predict,count,seed,g,check)
    lower=np.array([np.log10(bounds[n][0]) if n.startswith("k_") else bounds[n][0] for n in names])
    upper=np.array([np.log10(bounds[n][1]) if n.startswith("k_") else bounds[n][1] for n in names])
    def decode(q):return {**fixed,**{n:float(10**v) if n.startswith("k_") else float(v) for n,v in zip(names,q)}}
    endpoints,audit=solve_joint(times=t,observed=y,names=names,lower=lower,upper=upper,starts=starts,decode=decode,predict=predict,
        jacobian=None,max_nfev=config.fit.max_nfev,tolerance=config.solver.relative_tolerance,phase="M3 fixed-D conditional",
        cancel_check=cancel_check,progress=progress_callback)
    best=endpoints[0]
    from .production_stop import polish_joint
    best=polish_joint(best,phase='M3 fixed-D conditional',decode=decode,predict=predict,jacobian=None,observed=y)
    stats,warning=_fit_statistics(y,best.residuals,n_parameters=len(names))
    warning=tuple(warning)+tuple(_bound_warnings(names,best.coordinates,lower,upper))
    candidates=tuple(FitCandidate(e.index,{n:e.params[n] for n in names},e.sse,e.status,e.nfev,e.status>0,tuple(map(float,starts[e.index]))) for e in endpoints[:5])
    f=FitResult({n:best.params[n] for n in names},{n:"log" if n.startswith("k_") else "linear" for n in names},
        best.prediction,best.residuals,best.sse,exit_status=best.status,nfev=best.nfev,jacobian=best.jacobian,warnings=warning,candidates=candidates,
        diagnostics={"model_id":"M3_DUAL_HO_DK","analysis_dispatch":{"requested_model_id":"M3_DUAL_HO_DK","selected_model_id":"M3_DUAL_HO_DK",
            "m3_ran":True,"staged_m2_used":False,"inference_mode":"fixed_D_conditional"}},**stats)
    from pyecr.fitting.iteration_observation import observe_direct_forward
    observe_direct_forward()
    result=model.evaluate(t,**best.params,kernel_options=options)
    f=replace(f,diagnostics={**f.diagnostics,"analysis_artifacts":{"schema_version":1,"model_name":"M3_DUAL_HO_DK",
        "component_H":result.component_H.tolist(),"component_O":result.component_O.tolist(),"warnings":list(warning)}})
    uncertainty=estimate_local_uncertainty(f)
    check()
    annotated=record_randomness(uncertainty.annotated_fit_result,seed=seed,stochastic=count>10,
        algorithm='fixed-d-kinetic-grid-pcg64-full-linear-profile-v1',
        starts=[{'start_index':i,'coordinates':q.tolist()} for i,q in enumerate(starts)],deterministic_schedule_prefix=10)
    return _annotate(dataset,config,annotated,sources,extra={"starts":audit,"initial_profiles":profiles,
        "start_rule_version":"fixed-d-kinetic-grid-pcg64-full-linear-profile-v1","random_seed":seed,"selected_start_index":best.index,
        "parameter_bounds":{n:list(b) for n,b in bounds.items()},"observation_fixed":{n:v for n,v in fixed.items() if not n.startswith("D_")},
        "selected_model_id":"M3_DUAL_HO_DK","staged_m2_used":False,"ci_available":bool(uncertainty.reliable_separation and not any("PARAMETER_BOUND_HIT" in w for w in warning))})

def _annotate(dataset,config,fit,sources,extra):
    from pyecr.report.snapshot import dataset_sha256,_sha256_json
    effective_fixed={**config.model.fixed_parameters,**extra.get("observation_fixed",{})}
    record={"version":VERSION,"inference_mode":"fixed_D_conditional","physical_model_id":config.model_id.value,
        "sources":sources,"parent_fit_id":config.model_specific["fixed_d"].get("parent_fit_id"),
        "ci_scope":"conditional_on_D","propagates_D_uncertainty":False,
        "ci_conditions":{"fixed_D":{n:s["value_cm2_s"] for n,s in sources.items()},"other_fixed":effective_fixed,"geometry":config.model.geometry.to_dict()},
        "D_uncertainty_note":"Source uncertainty and same-data parent selection are recorded but not propagated.",
        "dataset_sha256":dataset_sha256(dataset),"config_sha256":_sha256_json(config.to_dict()),"free_parameter_count":len(fit.params),
        "parameter_roles":{**{n:"free" for n in fit.params},**{n:"fixed" for n in effective_fixed}},
        "fixed_parameters":effective_fixed,"declared_fixed_parameters":dict(config.model.fixed_parameters),"fit_eligible_count":dataset.processed_time.size,
        "original_raw_indices":np.flatnonzero(dataset.mask).tolist(),"fit_row_scope":"full_fit_eligible","no_analytic_limit_substitution":True,**extra}
    return replace(fit,diagnostics={**fit.diagnostics,"fixed_d":record},warnings=tuple(dict.fromkeys((*fit.warnings,
        "CONDITIONAL_CI: fixed-D/source uncertainty and parent plug-in selection are not propagated."))))
