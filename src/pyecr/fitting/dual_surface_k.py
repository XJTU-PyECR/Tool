"""M8 full-time joint inversion and complete input/report mode permutation."""
from pyecr.fitting.reproducibility import seeded_analysis_fit,record_randomness

from dataclasses import replace
import numpy as np
from scipy.optimize import lsq_linear
from pyecr.fitting.joint_trf import solve_joint
from pyecr.fitting.surface_k import _integer
from pyecr.fitting.single import _fit_statistics, _bound_warnings
from pyecr.fitting.base import FitResult, FitCandidate
from pyecr.fitting.solver_strategy import SolverOptions
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.models.surface_k import MODEL_VERSION, CONTRACT_VERSION, EXPOSURE_RULE
from pyecr.uncertainty.local import estimate_local_uncertainty

START_VERSION="m8-initial-anchor-pcg64-full-linear-profile-v1"
PARAMETERS=("k_1_cm_s","k_2_cm_s","A_1","A_2","y0")

def input_slot_name(name,perm):
    if name=="y0": return name
    slot=0 if "_1" in name else 1
    return name.replace(f"_{slot+1}",f"_{perm[slot]+1}")

def starts_for_joint(times,observed,names,fixed,bounds,initial,predict,n_starts,seed,g,check):
    """Versioned kinetic schedule; full-row linear observation initialization."""
    kinetic=names[:2]
    lo=np.array([np.log10(bounds[p][0]) for p in kinetic])
    hi=np.array([np.log10(bounds[p][1]) for p in kinetic])
    q0=np.array([np.log10(initial[p]) for p in kinetic])
    if np.any(q0<lo) or np.any(q0>hi) or not np.all(np.isfinite(q0)):
        raise ValueError("M8 kinetic initial values must lie in bounds")
    rng=np.random.default_rng(seed)
    draws=rng.uniform(np.nextafter(lo,hi),np.nextafter(hi,lo),size=(n_starts,2))
    positive=times[times>0]
    low=max(float(min(lo)),float(np.log10(.1/(g*times[-1]))))
    high=min(float(max(hi)),float(np.log10(10/(g*positive[0]))))
    if low>=high: low,high=float(min(lo)),float(max(hi))
    nodes=np.linspace(low,high,6)
    anchors=((4,2),(3,1),(2,0),(5,3),(4,1),(3,0),(1,3),(0,4))
    reversed_pair=q0[::-1]
    if np.any(reversed_pair<lo) or np.any(reversed_pair>hi): reversed_pair=(lo+hi)/2
    scheduled=[q0,reversed_pair]+[np.minimum(hi,np.maximum(lo,np.array([nodes[a],nodes[b]]))) for a,b in anchors]
    scheduled=scheduled[:n_starts]+[draws[i] for i in range(len(scheduled),n_starts)]
    starts,profiles=[],[]
    linear=names[2:]
    for i,qk in enumerate(scheduled):
        check()
        params={**fixed,**{p:float(10**v) for p,v in zip(kinetic,qk)},**{p:0. for p in linear}}
        if linear:
            base=predict(params)
            cols=[]
            for p in linear:
                changed={**params,p:1.}
                cols.append(predict(changed)-base)
            result=lsq_linear(np.column_stack(cols),observed-base,
                bounds=([bounds[p][0] for p in linear],[bounds[p][1] for p in linear]),tol=1e-10)
            params.update({p:float(v) for p,v in zip(linear,result.x)})
            profiles.append({"start_index":i,"method":"full-row bounded linear nuisance initialization",
                             "success":bool(result.success),"status":int(result.status),"rows":times.size})
        if i==0:
            for p in linear:
                if p in initial: params[p]=initial[p]
        q=np.array([np.log10(params[p]) if p in kinetic else params[p] for p in names])
        lower=np.array([np.log10(bounds[p][0]) if p in kinetic else bounds[p][0] for p in names])
        upper=np.array([np.log10(bounds[p][1]) if p in kinetic else bounds[p][1] for p in names])
        if np.any(q<lower) or np.any(q>upper) or not np.all(np.isfinite(q)):
            raise ValueError("joint observation initialization lies outside declared bounds")
        starts.append(q)
    return np.array(starts),profiles

def canonical_endpoint(endpoint,names,fixed,bounds,signs,label_semantics):
    params=endpoint.params
    perm=(0,1) if label_semantics=="attributed_species" or params["k_1_cm_s"]>=params["k_2_cm_s"] else (1,0)
    def input_name(name):
        return input_slot_name(name,perm)
    reported={p:params[input_name(p)] for p in PARAMETERS}
    free=tuple(p for p in PARAMETERS if input_name(p) in names)
    j=endpoint.jacobian[:,[names.index(input_name(p)) for p in free]]
    return reported,free,j,tuple(signs[i] for i in perm),perm,{
        p:bounds[input_name(p)] for p in free}, {p:reported[p] for p in PARAMETERS if input_name(p) in fixed}

@seeded_analysis_fit
def fit_dual_surface_k(dataset,config,*,cancel_check=None,progress_callback=None):
    config.validate()
    if config.model_id is not ModelID.M8_DUAL_K: raise ValueError("joint analytic fitter requires M8")
    from pyecr.fitting.coarse_to_fine import current_screening
    if current_screening() is not None: raise ValueError("M8 P60 screening Unsupported")
    if config.fit.loss!="linear" or SolverOptions.from_config(config).strategy!="trf":
        raise ValueError("M8 supports ideal_step/linear/TRF only")
    if set(config.model_specific)-{"dual_surface_k","local_solver"}:
        raise ValueError("M8 unsupported composition")
    options=config.model_specific.get("dual_surface_k",{})
    allowed={"n_starts","n_candidates","random_seed","label_semantics","label_mapping","label_evidence"}
    if set(options)-allowed: raise ValueError("Unknown M8 options or unsupported endpoint constraint")
    semantics=options.get("label_semantics","effective_modes")
    if semantics not in {"effective_modes","attributed_species"}: raise ValueError("Unknown M8 label semantics")
    mapping=options.get("label_mapping",{})
    if semantics=="attributed_species":
        evidence=options.get("label_evidence","")
        if set(mapping)!={"1","2"} or not all(isinstance(v,str) and v.strip() for v in mapping.values()) or mapping["1"]==mapping["2"] or not isinstance(evidence,str) or not evidence.strip():
            raise ValueError("Attributed M8 labels require two distinct explicit mappings and evidence")
    n_starts=_integer(options,"n_starts",20)
    keep=_integer(options,"n_candidates",5)
    seed=_integer(options,"random_seed",20260915)
    times,observed=dataset.processed_time,dataset.processed_signal
    if dataset.metadata.get("time_unit","s")!="s": raise ValueError("M8 requires seconds")
    if times.size<3 or np.any(times<0) or np.any(np.diff(times)<=0): raise ValueError("M8 needs all increasing nonnegative eligible rows")
    model=ModelFactory().create(config)
    fixed=dict(config.model.fixed_parameters)
    if {"k_1_cm_s","k_2_cm_s"}&set(fixed): raise ValueError("Both M8 kinetic parameters must be free")
    normalized=dataset.metadata.get("signal_type")=="normalized"
    requested=config.model.free_parameters or tuple(p for p in PARAMETERS if p not in fixed and (p!="y0" or not normalized))
    names=tuple(p for p in PARAMETERS if p in requested)
    if not {"k_1_cm_s","k_2_cm_s"}.issubset(names): raise ValueError("M8 requires both k to be free")
    for p,v in (("A_1",.1),("A_2",.1),("y0",0.)):
        if p not in names: fixed.setdefault(p,v)
    if normalized and ("y0" in names or fixed.get("y0")!=0): raise ValueError("Normalized M8 fixes y0=0")
    scale=max(float(np.ptp(observed)),float(np.max(np.abs(observed))),1.)
    defaults={"k_1_cm_s":(1e-10,1.),"k_2_cm_s":(1e-10,1.),"A_1":(0.,10*scale),"A_2":(0.,10*scale),
              "y0":(float(np.min(observed)-2*scale),float(np.max(observed)+2*scale))}
    bounds={p:config.model.parameter_bounds.get(p,defaults[p]) for p in names}
    initial={"k_1_cm_s":1e-3,"k_2_cm_s":1e-4,**config.model.initial_values}
    def check():
        if cancel_check: cancel_check()
    def predict(params):
        from pyecr.fitting.iteration_observation import observe_direct_forward
        observe_direct_forward()
        return model.predict(times,params)
    starts,profiles=starts_for_joint(times,observed,names,fixed,bounds,initial,predict,n_starts,seed,model.kernel.g_cm_inv,check)
    lower=np.array([np.log10(bounds[p][0]) if p.startswith("k_") else bounds[p][0] for p in names])
    upper=np.array([np.log10(bounds[p][1]) if p.startswith("k_") else bounds[p][1] for p in names])
    def decode(q): return {**fixed,**{p:float(10**v) if p.startswith("k_") else float(v) for p,v in zip(names,q)}}
    endpoints,audit=solve_joint(times=times,observed=observed,names=names,lower=lower,upper=upper,starts=starts,
        decode=decode,predict=predict,jacobian=lambda p:model.jacobian(times,p,names),max_nfev=config.fit.max_nfev,
        tolerance=config.solver.relative_tolerance,phase="M8",cancel_check=cancel_check,progress=progress_callback)
    best=endpoints[0]
    from .production_stop import polish_joint
    best=polish_joint(best,phase='M8',decode=decode,predict=predict,jacobian=lambda p:model.jacobian(times,p,names),observed=observed)
    params,free,j,signs,perm,reported_bounds,reported_fixed=canonical_endpoint(best,names,fixed,bounds,model.signs,semantics)
    from pyecr.fitting.iteration_observation import observe_direct_forward
    observe_direct_forward()
    _,input_components=model.evaluate(times,best.params)
    components=[input_components[i].tolist() for i in perm]
    warnings=list(_bound_warnings(names,best.coordinates,lower,upper))
    gap=abs(float(np.log10(params["k_1_cm_s"]/params["k_2_cm_s"])))
    total_a=params["A_1"]+params["A_2"]
    weak=[i for i in (1,2) if total_a==0 or params[f"A_{i}"]/total_a<.05]
    if gap<.05: warnings.append("IDENTIFIABILITY_NEAR_RATE: separate rates/amplitudes are not reliable")
    if weak: warnings.append(f"IDENTIFIABILITY_WEAK_PROCESS: mode(s) {weak}; corresponding kinetic CI unavailable")
    net=signs[0]*params["A_1"]+signs[1]*params["A_2"]
    cancellation=total_a==0 or abs(net)/total_a<1e-6
    if cancellation: warnings.append("NET_CHANGE_NORMALIZATION_UNAVAILABLE: preserve unitful transient; no [0,1] clipping")
    rates=[model.kernel.rates(params[f"k_{i}_cm_s"]) for i in (1,2)]
    if times[-1]<max(tau for _,tau in rates): warnings.append("IDENTIFIABILITY_TIME_COVERAGE: slow-mode tail not covered")
    labels=[mapping[str(i+1)] for i in perm] if semantics=="attributed_species" else ["Fast effective process","Slow effective process"]
    from pyecr.report.snapshot import dataset_sha256,_sha256_json
    record={"model_version":MODEL_VERSION,"contract_version":CONTRACT_VERSION,"method_version":"p69a-joint-surface-k-v1",
        "start_rule_version":START_VERSION,"random_seed":seed,"starts":audit,"initial_profiles":profiles,
        "selected_start_index":best.index,"selected_model_id":"M8_DUAL_K","selection_rule":"completed full-SSE/start-index; no automatic M7 replacement",
        "full_dims_cm":list(config.model.geometry.full_dims_cm),"active_dimensions":list(config.model.geometry.active_dimensions),
        "exposure_rule":EXPOSURE_RULE,"g_cm_inv":model.kernel.g_cm_inv,"input_parameters":best.params,
        "reported_parameters":params,"input_parameter_names":list(names),"reported_parameter_names":list(free),
        "reported_to_input":list(perm),"input_signs":list(model.signs),"signs":list(signs),"sign_source":config.model.step_type,
        "label_semantics":semantics,"label_mapping":dict(mapping),"label_evidence":options.get("label_evidence"),"component_labels":labels,
        "component_1":components[0],"component_2":components[1],"component_baseline":params["y0"],
        "parameter_bounds":{p:list(b) for p,b in reported_bounds.items()},"fixed_parameters":reported_fixed,
        "parameter_roles":{**{p:"free" for p in free},**{p:"fixed" for p in reported_fixed},
            **{p:"derived" for p in ("g_cm_inv","lambda_1_s_inv","lambda_2_s_inv","tau_1_surface_exchange_s","tau_2_surface_exchange_s")}},
        "free_parameter_count":len(free),"kinetic_free_count":2,"observation_free_count":len(free)-2,
        "near_rate_gap_decades":gap,"weak_processes":weak,"net_normalization_available":not cancellation,
        "signed_B":signs[0]*params["A_1"],"signed_C":signs[1]*params["A_2"],
        "bi":{"available":False,"reason":"D_not_estimated"},"ci_scope":"conditional_on_fixed_inputs",
        "ci_conditions":{"geometry":config.model.geometry.to_dict(),"input_fixed_parameters":fixed,"input_signs":list(model.signs),"label_semantics":semantics},
        "original_raw_indices":np.flatnonzero(dataset.mask).tolist(),"fit_eligible_count":times.size,
        "dataset_sha256":dataset_sha256(dataset),"config_sha256":_sha256_json(config.to_dict()),"fit_row_scope":"full_fit_eligible",
        "inference_mode":"analytic_surface_controlled","jacobian":"analytic log10 k / linear observation; complete reported column permutation",
        "units":{**{p:"cm/s" if p.startswith("k_") else str(dataset.metadata.get("signal_unit","signal")) for p in PARAMETERS},
                 "g_cm_inv":"cm^-1","tau_1_surface_exchange_s":"s","tau_2_surface_exchange_s":"s","lambda_1_s_inv":"s^-1","lambda_2_s_inv":"s^-1"}}
    for i,(rate,tau) in enumerate(rates,1): record.update({f"lambda_{i}_s_inv":rate,f"tau_{i}_surface_exchange_s":tau})
    candidates=[]
    for e in endpoints:
        reported,efree,_,es,ep,_,_=canonical_endpoint(e,names,fixed,bounds,model.signs,semantics)
        audit[e.index].update(reported_parameters=reported,reported_to_input=list(ep),reported_signs=list(es))
        if len(candidates)<keep:
            candidates.append(FitCandidate(e.index,{p:reported[p] for p in efree},e.sse,e.status,e.nfev,e.status>0,
                tuple(float(starts[e.index,names.index(input_slot_name(p,ep))]) for p in efree)))
    statistics,extra=_fit_statistics(observed,best.residuals,n_parameters=len(free))
    fit=FitResult({p:params[p] for p in free},{p:"log" if p.startswith("k_") else "linear" for p in free},
        best.prediction,best.residuals,best.sse,exit_status=best.status,nfev=best.nfev,jacobian=j,
        warnings=tuple(warnings)+extra,candidates=tuple(candidates),diagnostics={"model_id":"M8_DUAL_K","dual_surface_k":record},**statistics)
    uncertainty=estimate_local_uncertainty(fit)
    regular=bool(uncertainty.reliable_separation and gap>=.05 and not any("PARAMETER_BOUND_HIT" in w for w in warnings))
    available={p:regular and not any(p in {f"k_{i}_cm_s",f"A_{i}"} for i in weak) for p in free}
    record.update(ci_available=all(available.values()),ci_availability=available,
                  ci_unavailable_reason="rank/condition/bounds/near rate/weak amplitude require nonregular inference",
                  jacobian_rank=uncertainty.rank,condition_number=float(uncertainty.condition_number) if np.isfinite(uncertainty.condition_number) else None)
    for i in (1,2):
        interval=uncertainty.confidence_intervals.get(f"k_{i}_cm_s")
        if available.get(f"k_{i}_cm_s") and interval is not None and 0<interval.lower<=interval.upper and np.isfinite(interval.upper):
            record[f"tau_{i}_surface_exchange_ci_s"]=[1/(model.kernel.g_cm_inv*interval.upper),1/(model.kernel.g_cm_inv*interval.lower)]
        elif f"k_{i}_cm_s" in available: available[f"k_{i}_cm_s"]=False
    record["ci_available"]=all(available.values())
    check()
    if progress_callback: progress_callback(1.,"M8 whole-time joint fit complete")
    final=replace(uncertainty.annotated_fit_result,diagnostics={"model_id":"M8_DUAL_K","dual_surface_k":record})
    return record_randomness(final,seed=seed,stochastic=n_starts>10,algorithm=START_VERSION,
                             starts=[{'start_index':i,'coordinates':q.tolist()} for i,q in enumerate(starts)],
                             deterministic_schedule_prefix=10,unused_random_draws=n_starts<=10)
