"""Original unit/value declarations for an explicitly frozen Simulation input."""
from dataclasses import dataclass
from collections.abc import Mapping
from typing import ClassVar
import numpy as np
from pyecr.config import BoundaryConfig, _freeze_json
from pyecr.units import physical_unit, to_canonical, parameter_dimension, validate_pair
from .common import Contract, require, number, integer, vector, text, exact_fields

UNIT_INPUT_VERSION='simulation-input-units/1'


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationUnitInputs(Contract):
    KIND: ClassVar[str]='simulation_unit_inputs'
    parameters: Mapping[str,float]
    parameter_units: Mapping[str,str]
    dimensions: tuple[float,...]
    length_unit: str
    sampling_kind: str
    time_unit: str
    time_values: tuple[float,...]
    point_count: int | None
    boundary_input: Mapping[str,object]
    signal_type: str
    signal_unit: str
    noise_sd: float
    conversion_version: str=UNIT_INPUT_VERSION

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.conversion_version==UNIT_INPUT_VERSION,'Unknown Simulation input conversion version')
        require(isinstance(self.parameters,Mapping) and isinstance(self.parameter_units,Mapping),'Input parameters/units required')
        object.__setattr__(self,'parameters',_freeze_json({text(k,'parameter'):number(v,k) for k,v in self.parameters.items()},path='input parameters'))
        object.__setattr__(self,'parameter_units',_freeze_json({text(k,'parameter'):text(v,'parameter unit') for k,v in self.parameter_units.items()},path='parameter units'))
        object.__setattr__(self,'dimensions',vector(self.dimensions,'input full dimensions'))
        require(len(self.dimensions)==3,'Three complete input dimensions required')
        physical_unit('length',self.length_unit);physical_unit('time',self.time_unit)
        object.__setattr__(self,'time_values',vector(self.time_values,'input times'))
        require(self.sampling_kind in {'linear','log_with_zero','explicit'},'Unknown input sampling kind')
        if self.sampling_kind=='explicit':
            require(self.point_count is None and 2<=len(self.time_values)<=50000,'Explicit input count invalid; no truncation')
        else:
            n=integer(self.point_count,'input point count',minimum=2 if self.sampling_kind=='linear' else 3)
            require(n<=50000 and len(self.time_values)==2,'Sampling input count/endpoints invalid; no truncation')
        require(isinstance(self.boundary_input,Mapping),'Boundary input declarations required')
        object.__setattr__(self,'boundary_input',_freeze_json(self.boundary_input,path='boundary input'))
        validate_pair(self.signal_type,self.signal_unit);number(self.noise_sd,'input noise SD',nonnegative=True)

    def canonical_fields(self,model_id):
        """Physical input conversion only; never a forward model or RNG call."""
        from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
        desc=DEFAULT_MODEL_REGISTRY.get(model_id)
        require(set(self.parameters)==set(self.parameter_units)==set(desc.parameter_names),'Input parameter/unit inventory conflict')
        params={}
        for p in desc.parameters:
            dimension=parameter_dimension(p.unit);unit=self.parameter_units[p.name]
            if dimension:params[p.name]=to_canonical(self.parameters[p.name],dimension,unit)
            else:
                require(p.unit=='signal' and unit==self.signal_unit,'Amplitude/baseline uses the shared source signal unit')
                params[p.name]=self.parameters[p.name]
        dims=tuple(to_canonical(v,'length',self.length_unit) for v in self.dimensions)
        times=tuple(to_canonical(v,'time',self.time_unit) for v in self.time_values)
        if self.sampling_kind=='linear':times=tuple(np.linspace(times[0],times[1],self.point_count))
        elif self.sampling_kind=='log_with_zero':
            require(0<times[0]<times[1],'log_with_zero requires 0 < first positive time < stop')
            times=(0.,*np.geomspace(times[0],times[1],self.point_count-1))
        b=self.boundary_input;mode=b.get('mode')
        if mode=='ideal_step':
            exact_fields(b,('mode',),'ideal boundary input');boundary=BoundaryConfig()
        elif mode=='exponential_flush':
            exact_fields(b,('mode','residence_time','time_unit'),'flush boundary input')
            boundary=BoundaryConfig(mode=mode,flush_time_s=to_canonical(number(b['residence_time'],'flush time'),'time',b['time_unit']))
        elif mode=='measured_history':
            exact_fields(b,('mode','time','time_unit','values','signal_type','initial_value','final_value','value_unit',
                'resample_step','resample_unit','max_internal_points'),'measured boundary input')
            boundary=BoundaryConfig(mode=mode,
                measured_time_s=tuple(to_canonical(v,'time',b['time_unit']) for v in vector(b['time'],'history times')),
                measured_values=vector(b['values'],'history values'),measured_signal_type=b['signal_type'],
                measured_initial_value=number(b['initial_value'],'initial gas'),measured_final_value=number(b['final_value'],'final gas'),
                measured_value_unit=text(b['value_unit'],'gas unit'),
                resample_time_step_s=None if b['resample_step'] is None else to_canonical(number(b['resample_step'],'history step'),'time',b['resample_unit']),
                max_internal_points=integer(b['max_internal_points'],'internal point budget',minimum=1))
        else:raise ValueError('Unsupported input boundary mode')
        return dict(given_parameters=params,full_dims_cm=dims,time_s=times,boundary=boundary)

    def validate_request(self,request):
        converted=self.canonical_fields(request.model.model_id)
        require(converted['given_parameters']==dict(request.given_parameters)
            and converted['full_dims_cm']==request.model.geometry.full_dims_cm
            and converted['time_s']==request.time_s and converted['boundary']==request.boundary,
            'Simulation unit input/canonical request conflict')
        require((self.signal_type,self.signal_unit,self.noise_sd,self.sampling_kind)==
                (request.signal_type,request.signal_unit,request.noise_sd,request.sampling_kind),'Signal declaration/noise/sampling input conflict')
