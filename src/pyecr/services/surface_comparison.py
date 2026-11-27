"""Explicit same-data/same-observation comparison, without fitting or replacement."""
import numpy as np
from pyecr.models.ids import ModelID
from pyecr.report.snapshot import dataset_sha256, _sha256_json

def compare_surface_models(dataset,single_config,single_fit,dual_config,dual_fit):
    from pyecr.fitting.reproducibility import effective_config
    single_config=effective_config(single_config,single_fit)
    dual_config=effective_config(dual_config,dual_fit)
    reasons=[]
    if single_config.model_id is not ModelID.M7_SINGLE_K or dual_config.model_id is not ModelID.M8_DUAL_K:
        raise ValueError("Comparison requires M7 and M8")
    s=single_fit.diagnostics["surface_k"]
    d=dual_fit.diagnostics["dual_surface_k"]
    for cfg,fit,record in ((single_config,single_fit,s),(dual_config,dual_fit,d)):
        if record["dataset_sha256"]!=dataset_sha256(dataset) or record["config_sha256"]!=_sha256_json(cfg.to_dict()):
            reasons.append("dataset/config provenance mismatch")
        if fit.y_fit.size!=dataset.processed_time.size: reasons.append("fit-eligible rows differ")
    if single_config.model.geometry!=dual_config.model.geometry: reasons.append("geometry/exposure differs")
    if single_config.model.boundary_mode!=dual_config.model.boundary_mode or single_config.fit.loss!=dual_config.fit.loss:
        reasons.append("boundary/loss differs")
    single_baseline_free="baseline" in single_fit.params
    dual_baseline_free="y0" in dual_fit.params
    if single_baseline_free!=dual_baseline_free: reasons.append("baseline free/fixed constraint differs")
    elif single_baseline_free:
        if tuple(s["parameter_bounds"]["baseline"])!=tuple(d["parameter_bounds"]["y0"]): reasons.append("baseline bounds differ")
    elif s["fixed_parameters"]["baseline"]!=d["fixed_parameters"]["y0"]: reasons.append("fixed baseline differs")
    total_bounds=[0.,0.]
    total_free=False
    for i,sign in enumerate(d["signs"],1):
        name=f"A_{i}"
        free=name in dual_fit.params
        total_free|=free
        values=d["parameter_bounds"][name] if free else (d["fixed_parameters"][name],)*2
        signed=sorted(sign*float(v) for v in values)
        total_bounds=[a+b for a,b in zip(total_bounds,signed)]
    if ("amplitude" in single_fit.params)!=total_free: reasons.append("total amplitude constraint differs")
    else:
        single_bounds=s["parameter_bounds"]["amplitude"] if total_free else (s["fixed_parameters"]["amplitude"],)*2
        if not np.array_equal(single_bounds,total_bounds): reasons.append("signed total amplitude bounds/fixed value differ")
    eligible=not reasons
    chosen=None
    if eligible and np.isfinite(single_fit.AICc) and np.isfinite(dual_fit.AICc):
        chosen="M7_SINGLE_K" if single_fit.AICc<=dual_fit.AICc else "M8_DUAL_K"
    return {"version":"surface-model-comparison-v1","eligible":eligible,"reasons":reasons,
        "SSE":{"M7_SINGLE_K":single_fit.SSE,"M8_DUAL_K":dual_fit.SSE},
        "AICc":{"M7_SINGLE_K":single_fit.AICc if np.isfinite(single_fit.AICc) else None,"M8_DUAL_K":dual_fit.AICc if np.isfinite(dual_fit.AICc) else None},
        "free_counts":{"M7_SINGLE_K":len(single_fit.params),"M8_DUAL_K":len(dual_fit.params)},
        "statistical_preference":chosen,"replaces_fit":False,
        "interpretation":"A statistical preference under matched observation constraints is not physical correctness or species attribution."}
