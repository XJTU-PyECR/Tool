"""Versioned fit RNG provenance. No forward, objective, solver or search changes."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from numbers import Integral
from typing import Mapping
import hashlib,json
from functools import wraps

VERSION='fit-rng/1.0.0'
LEGACY_MESSAGE='Legacy fit lacks random-seed provenance; rerun fit to enable reproducible snapshot'
_job=ContextVar('pyecr_fit_rng_job',default=None)
SECTIONS={'M0_SINGLE_D':'single_ecr','M1_SINGLE_DK':'single_ecr','M2_DUAL_HO_D':'dual_ho',
          'M3_DUAL_HO_DK':'dual_ho','M7_SINGLE_K':'surface_k','M8_DUAL_K':'dual_surface_k','M5_GLOBAL':'global_ecr'}

def integer_seed(value):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
        raise ValueError('random_seed must be a nonnegative integer')
    return int(value)

def config_hash(config):
    return hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def bind_config(config, *, section=None):
    """Materialize the engine's existing default BEFORE initialization; never at capture."""
    from pyecr.config import AnalysisConfig,_freeze_json
    if not isinstance(config,AnalysisConfig):
        return config,{'version':VERSION,'allocation_phase':'validated_core_config_before_rng'}
    section=section or SECTIONS.get(config.model_id.value)
    if section is None:return config,{'version':VERSION,'allocation_phase':'unknown_backend'}
    opts=dict(config.model_specific.get(section,{}))
    supplied=config.model_specific.get('fit_random_seed',opts.get('random_seed'))
    source='explicit' if supplied is not None else 'automatic_versioned_engine_default'
    seed=integer_seed(supplied if supplied is not None else 20260916 if section=='global_ecr' else 20260915)
    if opts.get('random_seed')!=seed:
        config=replace(config,model_specific={**config.model_specific,section:{**opts,'random_seed':seed}})
    job={'version':VERSION,'allocation_phase':'fit_job_creation','seed_source':source,'seed_policy':'existing-engine-defaults-v1',
         'engine_section':section,'resolved_random_seed':seed,'effective_config_sha256':config_hash(config),'effective_config':config.to_dict()}
    prepared=_job.get()
    if prepared and prepared.get('effective_config_sha256')==job['effective_config_sha256']:job=prepared
    return config,_freeze_json(job,path='fit_random_job')

@contextmanager
def random_job_scope(job):
    token=_job.set(job)
    try:yield
    finally:_job.reset(token)

def core_job(seed,algorithm):
    return _job.get() or {'version':VERSION,'allocation_phase':'validated_core_config_before_rng',
                         'resolved_random_seed':integer_seed(seed),'seed_source':'validated_core_policy','algorithm':algorithm}

def seeded_analysis_fit(function):
    @wraps(function)
    def run(dataset,config,*args,**kwargs):
        config,job=bind_config(config)
        with random_job_scope(_job.get() or job):return function(dataset,config,*args,**kwargs)
    return run

def gui_seed_specific(specific,text):
    """Blank GUI means automatic, including after explicitly clearing an old seed."""
    if str(text or '').strip():return specific
    result=dict(specific);result.pop('fit_random_seed',None)
    for name in set(SECTIONS.values()):
        if name in result:
            opts=dict(result[name]);opts.pop('random_seed',None);result[name]=opts
    return result

def record_randomness(fit, *, seed, stochastic, algorithm, starts=(), job=None, **extra):
    """Record the seed used by the backend, checked against the pre-created job."""
    seed=integer_seed(seed);job=job or core_job(seed,algorithm)
    if job.get('resolved_random_seed',seed)!=seed:raise ValueError('Actual backend random seed differs from fit-job seed')
    record={**job,'version':VERSION,'stochastic':bool(stochastic),'random_seed':seed if stochastic else None,
            'configured_seed':seed,'bit_generator':'PCG64' if stochastic else None,'algorithm':algorithm,
            'initial_points':list(starts),**extra}
    return replace(fit,diagnostics={**fit.diagnostics,'fitting_randomness':record})

def fit_randomness(fit):
    record=fit.diagnostics.get('fitting_randomness')
    if isinstance(record,Mapping):
        if record.get('version')!=VERSION or not isinstance(record.get('stochastic'),bool):
            raise ValueError('Invalid fit randomness provenance version/state')
        stochastic=record['stochastic'];seed=record.get('random_seed')
        if stochastic:integer_seed(seed)
        elif seed is not None:raise ValueError('Deterministic fit must not claim a random seed')
        if not record.get('algorithm') or not record.get('allocation_phase'):
            raise ValueError('Fit randomness provenance lacks algorithm/allocation phase')
        return record
    # Compatibility only for actual, previously serialized backend seed fields.
    # Never infer from AnalysisConfig, solver ledgers or a caller's post-hoc seed.
    if fit.diagnostics.get('random_seed') is not None:
        return {'version':'legacy-recorded-seed','stochastic':True,'random_seed':integer_seed(fit.diagnostics['random_seed']),
                'algorithm':'legacy-backend-record','allocation_phase':'recorded_backend_seed'}
    for section in ('single_ecr','surface_k','dual_surface_k','fixed_d'):
        payload=fit.diagnostics.get(section,{})
        if isinstance(payload,Mapping) and payload.get('random_seed') is not None:
            return {'version':'legacy-recorded-seed','stochastic':True,'random_seed':integer_seed(payload['random_seed']),
                    'algorithm':section,'allocation_phase':'recorded_backend_seed'}
    return None

def require_reproducibility(fit, explicit=None):
    record=fit_randomness(fit)
    if record is None:raise RuntimeError(LEGACY_MESSAGE)
    seed=record['random_seed']
    if explicit is not None and integer_seed(explicit)!=seed:
        raise ValueError('Snapshot/project seed must equal the actual fit seed; it cannot be replaced after fitting')
    return seed

def reproducibility_issue(fit):
    if fit is None:return ''
    try:require_reproducibility(fit)
    except (ValueError,RuntimeError) as exc:return str(exc)
    return ''

def effective_config(requested,fit):
    from pyecr.config import AnalysisConfig
    record=fit_randomness(fit) or {};payload=record.get('effective_config')
    if not payload:return requested
    used=AnalysisConfig.from_dict(payload)
    def physical(config):
        data=config.to_dict();specific=data['model_specific'];specific.pop('fit_random_seed',None)
        for opts in specific.values():
            if isinstance(opts,dict):opts.pop('random_seed',None)
        return data
    if physical(requested)!=physical(used):raise ValueError('Fit provenance effective config differs from scientific inputs')
    section=record.get('engine_section');supplied=requested.model_specific.get('fit_random_seed',requested.model_specific.get(section,{}).get('random_seed'))
    if supplied is not None and integer_seed(supplied)!=record['configured_seed']:
        raise ValueError('Config seed differs from actual fit-job seed')
    return requested if used==requested else used
