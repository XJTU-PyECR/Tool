"""Explicit, source-bound fixed diffusion configuration; not an analytic limit."""
from dataclasses import replace
from math import isfinite
from collections.abc import Mapping
from pyecr.models.ids import ModelID
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.fitting.base import FitResult
from pyecr.services.guided_refit import fit_result_id

VERSION="p69b-fixed-d-conditional-v1"
KINETICS={ModelID.M1_SINGLE_DK:("D_cm2_s",),ModelID.M3_DUAL_HO_DK:("D_H_cm2_s","D_O_cm2_s")}

def _sources(model_id,sources,parent_payload=None,parent_id=None):
    if model_id not in KINETICS or set(sources)!=set(KINETICS[model_id]):
        raise ValueError("Fixed-D entry requires exactly the M1 D or both M3 D sources")
    parent=None
    if parent_payload is not None:
        parent=FitResult.from_dict(parent_payload)
        if fit_result_id(parent)!=parent_id: raise ValueError("Fixed-D parent FitResult content ID mismatch")
    result={}
    for name,source in sources.items():
        if not isinstance(source,Mapping): raise ValueError("D source must be a record")
        value=source.get("value_cm2_s")
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not isfinite(value) or value<=0:
            raise ValueError("Fixed D must be a positive finite physical value")
        if source.get("unit")!="cm^2/s": raise ValueError("Fixed D source unit must be cm^2/s")
        kind=source.get("kind")
        if kind not in {"manual","external_measurement","literature","parent_fit"}: raise ValueError("Unknown D value source kind")
        reference=source.get("reference")
        uncertainty=source.get("uncertainty")
        if not isinstance(reference,str) or not reference.strip(): raise ValueError("Fixed D requires a source reference/protocol")
        if not isinstance(uncertainty,str) or not uncertainty.strip(): raise ValueError("Describe fixed-D uncertainty, including explicitly unknown")
        independence=source.get("independence","unknown")
        if independence not in {"unknown","independent_declared","same_data_plugin"}: raise ValueError("Unknown D source independence")
        if kind=="parent_fit":
            if parent is None or name not in parent.params or float(value)!=parent.params[name]:
                raise ValueError("Parent-fit D must match the bound parent estimate exactly")
            if independence!="same_data_plugin": raise ValueError("Parent-fit D is a same-data plug-in, not an independent measurement")
        result[name]={"value_cm2_s":float(value),"unit":"cm^2/s","kind":kind,"reference":reference,
                      "uncertainty":uncertainty,"independence":independence}
    return result

def with_fixed_diffusion(config,sources,*,parent_fit=None):
    """Construct an explicit child config, preserving its observation policy."""
    payload=None if parent_fit is None else parent_fit.to_dict()
    parent_id=None if parent_fit is None else fit_result_id(parent_fit)
    sources=_sources(config.model_id,sources,payload,parent_id)
    fixed={**config.model.fixed_parameters,**{n:s["value_cm2_s"] for n,s in sources.items()}}
    names=config.model.free_parameters or tuple(p.name for p in DEFAULT_MODEL_REGISTRY.get(config.model_id).parameters if p.default_free)
    free=tuple(n for n in names if n not in fixed)
    model=replace(config.model,free_parameters=free,fixed_parameters=fixed,
                  initial_values={n:v for n,v in config.model.initial_values.items() if n not in sources})
    record={"version":VERSION,"sources":sources,"parent_fit_id":parent_id,"parent_fit":payload,
            "ci_scope":"conditional_on_D","propagates_D_uncertainty":False}
    child=replace(config,model=model,model_specific={**{k:v for k,v in config.model_specific.items() if k!="fixed_d_sources"},"fixed_d":record})
    DEFAULT_MODEL_REGISTRY.validate_config(child.model)
    child.validate()
    return child

def validate_fixed_diffusion(config):
    record=config.model_specific.get("fixed_d")
    if not isinstance(record,Mapping) or record.get("version")!=VERSION:
        raise ValueError("Explicit fixed-D entry requires its versioned source record")
    sources=_sources(config.model_id,record.get("sources",{}),record.get("parent_fit"),record.get("parent_fit_id"))
    for name,source in sources.items():
        if config.model.fixed_parameters.get(name)!=source["value_cm2_s"] or name in config.model.free_parameters:
            raise ValueError("Fixed-D source/config free-fixed mismatch")
    if record.get("ci_scope")!="conditional_on_D" or record.get("propagates_D_uncertainty") is not False:
        raise ValueError("Fixed-D CI must be explicitly conditional without D uncertainty propagation")
    return sources
