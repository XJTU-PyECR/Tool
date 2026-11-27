"""Simulation-only drafts and immutable session records; no forward equations."""
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from math import isfinite
import json
from uuid import uuid4

import numpy as np
from pyecr.config import BoundaryConfig, ModelConfig, SampleGeometry, SolverConfig
from pyecr.contracts import SimulationRequest, SimulationResult, SourceIdentity, canonical_sha256
from pyecr.contracts.simulation import SIMULATION_POINT_BUDGET
from pyecr.models.ids import ModelID
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.services.simulation_service import simulation_service_support
from pyecr.units import physical_unit, parameter_dimension
from pyecr.contracts.simulation_units import SimulationUnitInputs

MATRIX={row['model_id']:row for row in simulation_service_support()['model_matrix']}
UI_POINT_BUDGET=SIMULATION_POINT_BUDGET
MAX_PENDING_RUNS=8
CANCEL_NOTICE='Cancel requested: publication blocked. Stops at the next checkpoint; an in-flight production kernel/ODE call may need to return first.'


def numeric_sequence(text):
    values=json.loads(text) if str(text).strip().startswith('[') else str(text).replace(',',' ').split()
    if any(isinstance(v,bool) or v is None for v in values):
        raise ValueError('Time/history sequences require numeric observations, not boolean or null values')
    return tuple(float(v) for v in values)


def request_signature(request):
    wire=request.to_dict();wire.pop('job_id')
    return canonical_sha256(wire)


def now():return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelDraft:
    parameters: dict[str,str]
    boundary_mode: str='ideal_step'
    step_type: str='hydration'
    sign_H: str='-1'
    sign_O: str='1'
    flush_time_s: str='80'
    history_time: str='0,100,400,12000'
    history_value: str='0.1,0.4,1,1'
    gas_type: str='pO2'
    initial_gas: str='0.1'
    final_gas: str='1'
    gas_unit: str='bar'
    internal_step: str=''
    internal_budget: str='200000'
    parameter_units: dict[str,str]=field(default_factory=dict)
    flush_time_unit: str='s'
    history_time_unit: str='s'
    internal_step_unit: str='s'


@dataclass
class SimulationDraft:
    model_id: str='M0_SINGLE_D'
    models: dict[str,ModelDraft]=field(default_factory=dict)
    dims: tuple[str,...]=('0.1','0.2','0.3')
    active: tuple[bool,...]=(True,False,False)
    signal_type: str='normalized'
    signal_unit: str='1'
    sampling: str='linear'
    start: str='0'
    stop: str='12000'
    count: str='240'
    first_positive: str='0.1'
    explicit_times: str='0,1,10,100,1000,12000'
    relative_tolerance: str='1e-10'
    absolute_tolerance: str='1e-8'
    max_terms: str='512'
    name: str='Simulation'
    noise_sd: str='0'
    seed: str='12345'
    sequence_number: str='0'
    parameter_origin: str='manual'
    copied_from: SourceIdentity | None=None
    source_note: str='Manual given parameters; no imported observations or Fit estimates.'
    revision: int=0
    length_unit: str='cm'
    time_unit: str='s'
    signal_custom_unit: str=''

    @property
    def current(self):
        if self.model_id not in MATRIX:raise ValueError('Unsupported Simulation model: '+self.model_id)
        if self.model_id not in self.models:
            desc=DEFAULT_MODEL_REGISTRY.get(self.model_id)
            self.models[self.model_id]=ModelDraft({p.name:str(p.default_initial) for p in desc.parameters})
        return self.models[self.model_id]

    def edited(self):
        self.revision+=1
        if self.copied_from is not None:
            self.parameter_origin='manual'
            self.source_note='Manually edited Simulation draft, initially copied from applied Analysis configuration; parent identity retained, no Fit reliability transferred.'

    def select_model(self,model_id):
        if model_id not in MATRIX:raise ValueError('Unsupported Simulation model: '+model_id)
        if model_id==self.model_id:
            self.current
            return
        self.model_id=model_id;self.current;self.edited()

    def build_request(self,job_id='draft-review'):
        """Freeze once from source-unit inputs. Never rewrite draft values."""
        m=self.current;desc=DEFAULT_MODEL_REGISTRY.get(self.model_id)
        required=[('length',self.length_unit,'cm'),('time',self.time_unit,'s')]
        required.extend((dimension,m.parameter_units.get(p.name,p.unit),p.unit)
            for p in desc.parameters if (dimension:=parameter_dimension(p.unit)))
        if m.boundary_mode=='exponential_flush':required.append(('time',m.flush_time_unit,'s'))
        if m.boundary_mode=='measured_history':
            required.append(('time',m.history_time_unit,'s'))
            if m.internal_step.strip():required.append(('time',m.internal_step_unit,'s'))
        converted=any(physical_unit(dimension,unit).key!=canonical for dimension,unit,canonical in required)
        # Validate every active unit even if an earlier one needs conversion.
        for dimension,unit,_ in required:physical_unit(dimension,unit)
        if not converted:return self._build_canonical_request(job_id)
        row=MATRIX[self.model_id]
        if m.boundary_mode not in row['boundary_modes']:
            raise ValueError('Unsupported Simulation boundary for '+self.model_id+': '+m.boundary_mode)
        inputs=self._unit_inputs();values=inputs.canonical_fields(self.model_id)
        mode=row['surface_modes'][0];sd=float(self.noise_sd)
        return SimulationRequest(schema_version=3,input_units=inputs,name=self.name,sequence_number=int(self.sequence_number),
            noise_sd=sd,seed=int(self.seed) if sd>0 else None,rng_algorithm='PCG64' if sd>0 else None,
            model=ModelConfig(ModelID(self.model_id),SampleGeometry(values['full_dims_cm'],self.active),surface_mode=mode,
                boundary_mode=m.boundary_mode,step_type=m.step_type,
                sign_H=float(m.sign_H) if m.step_type=='custom' else None,sign_O=float(m.sign_O) if m.step_type=='custom' else None),
            boundary=values['boundary'],solver_config=SolverConfig(surface_mode=mode,relative_tolerance=float(self.relative_tolerance),
                absolute_tolerance=float(self.absolute_tolerance),max_terms=int(self.max_terms)),
            given_parameters=values['given_parameters'],time_s=values['time_s'],signal_type=self.signal_type,signal_unit=self.signal_unit,
            sampling_kind=self.sampling,job_id=job_id,parameter_origin=self.parameter_origin,copied_from=self.copied_from)

    def _unit_inputs(self):
        m=self.current;desc=DEFAULT_MODEL_REGISTRY.get(self.model_id)
        params={}
        for p in desc.parameters:
            try:params[p.name]=float(m.parameters[p.name])
            except (KeyError,TypeError,ValueError) as exc:raise ValueError('Given parameter '+p.name+': enter an explicit finite physical value') from exc
        b=dict(mode=m.boundary_mode)
        if m.boundary_mode=='exponential_flush':b.update(residence_time=float(m.flush_time_s),time_unit=m.flush_time_unit)
        elif m.boundary_mode=='measured_history':b.update(time=numeric_sequence(m.history_time),time_unit=m.history_time_unit,
            values=numeric_sequence(m.history_value),signal_type=m.gas_type,initial_value=float(m.initial_gas),final_value=float(m.final_gas),
            value_unit=m.gas_unit,resample_step=None if not m.internal_step.strip() else float(m.internal_step),
            resample_unit=m.internal_step_unit,max_internal_points=int(m.internal_budget))
        if self.sampling=='explicit':times=numeric_sequence(self.explicit_times);count=None
        else:
            count=int(self.count)
            if not 2<=count<=UI_POINT_BUDGET:raise ValueError(f'Simulation page point budget: 2..{UI_POINT_BUDGET}; no clipping/downsampling is performed')
            times=(float(self.start if self.sampling=='linear' else self.first_positive),float(self.stop))
        return SimulationUnitInputs(parameters=params,parameter_units={p.name:
            m.parameter_units.get(p.name,p.unit) if parameter_dimension(p.unit) else self.signal_unit for p in desc.parameters},
            dimensions=tuple(float(v) for v in self.dims),length_unit=self.length_unit,
            sampling_kind=self.sampling,time_unit=self.time_unit,time_values=times,point_count=count,
            boundary_input=b,signal_type=self.signal_type,signal_unit=self.signal_unit,noise_sd=float(self.noise_sd))

    def _build_canonical_request(self,job_id):
        m=self.current;row=MATRIX[self.model_id]
        if m.boundary_mode not in row['boundary_modes']:
            raise ValueError(f'Unsupported: {row["menu_alias"]} ({self.model_id}) supports only '+', '.join(row['boundary_modes'])+'; no dual/k-only gas convolution is implemented.')
        desc=DEFAULT_MODEL_REGISTRY.get(self.model_id)
        # Whitelist the current registered fields. A hidden old model value can
        # never silently satisfy another model's missing parameter.
        params={}
        for name in desc.parameter_names:
            try:params[name]=float(m.parameters[name])
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError('Given parameter '+name+': enter an explicit finite physical value') from exc
        options={}
        if m.boundary_mode=='exponential_flush':options['flush_time_s']=float(m.flush_time_s)
        elif m.boundary_mode=='measured_history':
            options=dict(measured_time_s=numeric_sequence(m.history_time),measured_values=numeric_sequence(m.history_value),
                measured_signal_type=m.gas_type,measured_initial_value=float(m.initial_gas),
                measured_final_value=float(m.final_gas),measured_value_unit=m.gas_unit,
                resample_time_step_s=None if not m.internal_step.strip() else float(m.internal_step),max_internal_points=int(m.internal_budget))
        boundary=BoundaryConfig(mode=m.boundary_mode,**options)
        if self.sampling=='explicit':times=numeric_sequence(self.explicit_times)
        else:
            n=int(self.count)
            if not 2<=n<=UI_POINT_BUDGET:raise ValueError(f'Simulation page point budget: 2..{UI_POINT_BUDGET}; no clipping/downsampling is performed')
            if self.sampling=='linear':times=tuple(np.linspace(float(self.start),float(self.stop),n))
            elif self.sampling=='log_with_zero':
                first,last=float(self.first_positive),float(self.stop)
                if n<3 or not (0<first<last) or not all(isfinite(v) for v in (first,last)):
                    raise ValueError('log_with_zero requires >=3 points and 0 < first positive time < stop')
                times=(0.,*np.geomspace(first,last,n-1))
            else:raise ValueError('Unknown time sampling mode')
        if len(times)>UI_POINT_BUDGET:raise ValueError(f'Simulation page point budget is {UI_POINT_BUDGET}; no observations were truncated')
        mode=row['surface_modes'][0]
        sd=float(self.noise_sd)
        return SimulationRequest(schema_version=2, name=self.name, sequence_number=int(self.sequence_number),
            noise_sd=sd,seed=int(self.seed) if sd>0 else None,rng_algorithm='PCG64' if sd>0 else None,
            model=ModelConfig(ModelID(self.model_id),SampleGeometry(tuple(float(v) for v in self.dims),self.active),
                surface_mode=mode,boundary_mode=boundary.mode,step_type=m.step_type,
                sign_H=float(m.sign_H) if m.step_type=='custom' else None,sign_O=float(m.sign_O) if m.step_type=='custom' else None),
            boundary=boundary,solver_config=SolverConfig(surface_mode=mode,relative_tolerance=float(self.relative_tolerance),
                absolute_tolerance=float(self.absolute_tolerance),max_terms=int(self.max_terms)),
            given_parameters=params,time_s=times,signal_type=self.signal_type,signal_unit=self.signal_unit,
            sampling_kind=self.sampling,job_id=job_id,parameter_origin=self.parameter_origin,copied_from=self.copied_from)


@dataclass(frozen=True)
class SimulationRun:
    request: SimulationRequest
    draft_signature: str
    source_note: str
    created_at: str
    status: str='queued'
    message: str='Queued frozen Simulation request'
    result: SimulationResult | None=None
    traceback_text: str=''
    finished_at: str | None=None
    progress: float=0.

    @property
    def job_id(self):return self.request.job_id


class SimulationWorkspace:
    """Session-only history, independent of Analysis and project persistence."""
    def __init__(self):
        self.draft=SimulationDraft();self._runs={};self.selected_id=None;self.last_complete_id=None

    @property
    def history(self):return tuple(self._runs.values())
    @property
    def selected(self):return self._runs.get(self.selected_id)
    @property
    def active_ids(self):return tuple(r.job_id for r in self.history if r.status in {'queued','running','cancelling'})
    @property
    def active_id(self):return next(reversed(self.active_ids),None)

    def prepare(self):
        if len(self.active_ids)>=MAX_PENDING_RUNS:raise RuntimeError(f'Simulation pending-job budget is {MAX_PENDING_RUNS}; finish or cancel a job')
        request=self.draft.build_request('simulation-'+uuid4().hex)
        run=SimulationRun(request,request_signature(request),self.draft.source_note,now())
        self._runs[run.job_id]=run;self.selected_id=run.job_id
        return run

    def update(self,key,**fields):
        self._runs[key]=replace(self._runs[key],**fields)
        return self._runs[key]

    def select(self,key):
        if key not in self._runs:raise ValueError('Unknown Simulation record')
        self.selected_id=key

    def restore_result(self,result):
        if not isinstance(result,SimulationResult) or not result.state.ready:raise ValueError('complete Simulation result required')
        key=result.request.job_id;existing=self._runs.get(key)
        if existing is not None:
            if existing.result is None or existing.result.content_sha256!=result.content_sha256:
                raise ValueError('Simulation job identity collision; existing history preserved')
            self.selected_id=key
            return existing
        run=SimulationRun(result.request,request_signature(result.request),
            'Explicitly loaded verified Simulation bundle; no forward/RNG/Fit executed. Created time below is this load time.',
            now(),status=result.state.status,message='Loaded complete simulated observations; warnings retained: '+'; '.join(result.state.warnings),
            result=result,finished_at=now(),progress=1.)
        self._runs[key]=run;self.selected_id=key;self.last_complete_id=key
        return run

    def complete(self,key,result):
        run=self._runs[key]
        if run.status=='cancelling':return self.finish_cancel(key)
        if not isinstance(result,SimulationResult) or result.request!=run.request or result.request_sha256!=run.request.content_sha256:
            raise ValueError('Simulation result/request identity conflict; publication refused')
        self.update(key,status=result.state.status,result=result,
            message='; '.join((*result.state.blocking_reasons,*result.state.warnings)) or 'Simulated observations generated; statistical reliability not evaluated.',
            finished_at=now(),progress=1. if result.state.ready else run.progress)
        if result.state.ready:self.last_complete_id=key

    def finish_cancel(self,key):
        self.update(key,status='cancelled',message='Cancelled; no result published. Prior records retained.',result=None,finished_at=now())

    def fail(self,key,message,traceback_text=''):
        self.update(key,status='error',message=str(message),result=None,traceback_text=traceback_text,finished_at=now())

    def copy_analysis(self,session):
        config=session.current_analysis_config
        if config is None or session.dataset is None or session.preprocessing is None:
            raise ValueError('Copy requires an applied Analysis configuration, dataset and preprocessing identity')
        if config.model_id.value not in MATRIX:raise ValueError('Applied Analysis model is not supported by Simulation')
        identity=SourceIdentity.capture(session.dataset,session.preprocessing.config,config=config)
        desc=DEFAULT_MODEL_REGISTRY.get(config.model_id)
        # Fit estimates are deliberately not inspected. Missing configured
        # values remain blank for explicit user entry rather than invented.
        values={**config.model.initial_values,**config.model.fixed_parameters}
        m=ModelDraft({p.name:str(values[p.name]) if p.name in values else '' for p in desc.parameters},
            boundary_mode=config.boundary.mode,step_type=config.model.step_type,
            sign_H=str(config.model.sign_H if config.model.sign_H is not None else -1.),
            sign_O=str(config.model.sign_O if config.model.sign_O is not None else 1.))
        b=config.boundary
        if b.mode=='exponential_flush':m.flush_time_s=str(b.flush_time_s)
        if b.mode=='measured_history':
            m.history_time=json.dumps(b.measured_time_s);m.history_value=json.dumps(b.measured_values)
            m.gas_type=b.measured_signal_type;m.initial_gas=str(b.measured_initial_value);m.final_gas=str(b.measured_final_value)
            m.gas_unit=b.measured_value_unit;m.internal_step='' if b.resample_time_step_s is None else str(b.resample_time_step_s)
            m.internal_budget=str(b.max_internal_points)
        d=self.draft;d.model_id=config.model_id.value;d.models[d.model_id]=m
        d.dims=tuple(format(v,'.17g') for v in config.model.geometry.full_dims_cm);d.active=config.model.geometry.active_dimensions
        d.length_unit='cm'  # Analysis stores full dimensions and kinetics in canonical units.
        d.signal_type=session.dataset.metadata.get('signal_type','normalized');d.signal_unit=session.dataset.metadata.get('signal_unit','1')
        d.relative_tolerance=str(config.solver.relative_tolerance);d.absolute_tolerance=str(config.solver.absolute_tolerance);d.max_terms=str(config.solver.max_terms)
        d.parameter_origin='analysis_config';d.copied_from=identity;d.revision+=1
        d.source_note='Explicitly copied applied Analysis configuration: configured initial/fixed values, not fitted estimates. Simulation time sampling retained. Source: '+str(session.dataset.metadata.get('source_path','in-memory dataset'))
        return tuple(p.name for p in desc.parameters if p.name not in values)


def complete_evidence(workspace):
    d=workspace.draft;run=workspace.selected
    lines=['Simulation — simulated observations; session-only records.',d.source_note,
        'Draft changes do not recompute stored runs or modify Analysis. Full scientific arrays below; no resampling, CI, R² or Fit state.',
        'Pending jobs: '+str(workspace.active_ids)+'; last complete: '+str(workspace.last_complete_id)]
    try:
        draft=d.build_request();lines+=['Draft signature: '+request_signature(draft),draft.to_json(indent=2)]
        if run:lines+=['Draft '+('matches' if run.draft_signature==request_signature(draft) else 'differs from')+' selected stored request.']
    except Exception as exc:lines+=['Draft invalid: '+str(exc)]
    if run:
        lines += [f'Selected {run.job_id} · {run.status}',run.message,run.source_note,
            'Created '+run.created_at+'; finished '+str(run.finished_at),
            'Immutable stored request:\n'+run.request.to_json(indent=2)]
        if run.result is not None:lines+=['Stored simulation result (all arrays):\n'+run.result.to_json(indent=2)]
        if run.traceback_text:lines+=[run.traceback_text]
    else:lines+=['No Simulation result has been generated.']
    return '\n\n'.join(lines)
