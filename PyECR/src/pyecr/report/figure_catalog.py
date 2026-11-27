"""Portable immutable figure descriptions; no solver and no GUI dependency."""
from dataclasses import dataclass,asdict
import hashlib,json,re
import numpy as np
from pyecr.project.repository import _encode_special_floats,_decode_special_floats

VERSION='pyecr-figures/1'

def freeze(value):
    return json.dumps(_encode_special_floats(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def thaw(value):return _decode_special_floats(json.loads(value))
def digest(value):return hashlib.sha256(freeze(value).encode('utf-8')).hexdigest()
def safe_id(value):
    name=re.sub('[^a-zA-Z0-9_.-]+','_',str(value)).strip('._')[:64]
    return (name or 'item')+'_'+digest(str(value))[:10]

@dataclass(frozen=True)
class FigureSeries:
    label: str
    x: tuple
    y: tuple
    series_id: str=''
    kind: str='line'
    sampling: str='scientific'
    color: str | None=None
    def __post_init__(self):
        object.__setattr__(self,'x',tuple(float(v) for v in self.x))
        object.__setattr__(self,'y',tuple(float(v) for v in self.y))
        if self.kind not in ('line','scatter','hline','bar'):raise ValueError('Unknown figure series kind')
        if self.kind!='hline' and len(self.x)!=len(self.y):raise ValueError('Figure series arrays must align')
        if self.kind=='hline' and len(self.y)!=1:raise ValueError('Reference line needs one y value')
    def to_dict(self):return asdict(self)

@dataclass(frozen=True)
class FigureDescriptor:
    figure_id: str
    module: str
    result_type: str
    source_id: str
    title: str
    plot_id: str=''
    xlabel: str=''
    ylabel: str=''
    x_unit: str='1'
    y_unit: str='1'
    series: tuple=()
    view_state: str | None=None
    xscale: str='linear'
    yscale: str='linear'
    view_log_coordinates: tuple=(False,False)
    use_view_ranges: bool=True
    renderer: str='series'
    payload: str='{}'
    unavailable_reason: str | None=None
    formats: tuple=('png','svg','pdf')
    def __post_init__(self):
        object.__setattr__(self,'series',tuple(self.series))
        if not re.fullmatch('[a-zA-Z0-9_.-]+',self.figure_id):raise ValueError('Unsafe figure_id')
        if self.renderer not in ('series','objective_map'):raise ValueError('Unknown figure renderer')
        if self.xscale not in ('linear','log') or self.yscale not in ('linear','log'):raise ValueError('Unknown axis transform')
        if self.view_state is not None:
            from pyecr.plotting.view_state import freeze_views
            object.__setattr__(self,'view_state',freeze_views(self.view_state))
        if not self.series and self.renderer=='series' and not self.unavailable_reason:
            object.__setattr__(self,'unavailable_reason','No captured data for this figure.')
        if self.series and not self.unavailable_reason and not any(np.isfinite(s.y).any() and (s.kind=='hline' or np.isfinite(s.x).any()) for s in self.series if s.kind!='hline'):
            object.__setattr__(self,'unavailable_reason','No finite captured values for this figure.')
    @property
    def available(self):return self.unavailable_reason is None
    def to_dict(self):
        return {**asdict(self),'series':[s.to_dict() for s in self.series]}
    @classmethod
    def from_dict(cls,value):
        v=dict(value);v['series']=tuple(FigureSeries(**s) for s in v['series'])
        v['formats']=tuple(v['formats']);v['view_log_coordinates']=tuple(v['view_log_coordinates'])
        return cls(**v)

@dataclass(frozen=True)
class FigureSource:
    module: str
    result_type: str
    payload: str
    scope: str='current'
    label: str=''
    view_state: str | None=None
    display_options: str='{}'
    def __post_init__(self):
        if self.scope not in ('current','historical'):raise ValueError('Select current or explicit historical evidence')
        if self.module not in ('Analysis','Batch','Global','Pair','DCT','Simulation'):raise ValueError('Unknown figure source module')
        if self.view_state is not None:
            from pyecr.plotting.view_state import freeze_views
            object.__setattr__(self,'view_state',freeze_views(self.view_state))
        options=thaw(self.display_options)
        if not isinstance(options,dict) or set(options)-{'version','dct_physical','component_scale','component_view','sensitivity_normalized'}:raise ValueError('Unknown figure display options')
        if options and options.get('version')!=1:raise ValueError('Unknown figure display version')
        for key in ('dct_physical','sensitivity_normalized'):
            if key in options and type(options[key]) is not bool:raise ValueError('Figure display flags must be boolean')
        for key,allowed in (('component_view',('Overall','component_1','component_2','Comparison')),('component_scale',('Independent shape','Common signed scale'))):
            if key in options and options[key] not in allowed:raise ValueError('Unknown figure '+key)
        # Adapters validate original versioned contracts, not merely a JSON hash.
        from pyecr.report.figure_sources import validate_source
        validate_source(self)
    @property
    def result_sha256(self):return hashlib.sha256(self.payload.encode('utf-8')).hexdigest()
    @property
    def source_id(self):return self.module.lower()+'_'+self.result_sha256[:20]
    def to_dict(self):return dict(version=VERSION,module=self.module,result_type=self.result_type,payload=thaw(self.payload),scope=self.scope,label=self.label,view_state=self.view_state,display_options=thaw(self.display_options),source_id=self.source_id,result_sha256=self.result_sha256)
    @classmethod
    def from_dict(cls,value):
        if value.get('version')!=VERSION:raise ValueError('Unknown figure source version')
        obj=cls(value['module'],value['result_type'],freeze(value['payload']),value['scope'],value['label'],value['view_state'],freeze(value.get('display_options',{})))
        if obj.source_id!=value['source_id'] or obj.result_sha256!=value['result_sha256']:raise ValueError('Figure source identity conflict')
        return obj

def capture_source(result,*,entries=(),template=None,interpretations=None,scope='current',label='',view_state=None):
    """Detach from drafts; keep original contract plus bound source observations."""
    from pyecr.report.snapshot import AnalysisSnapshot,_dataset_to_dict
    from pyecr.services.batch_analysis import BatchAnalysisResult,GlobalAnalysisResult
    from pyecr.services.pair_design_state import PairDesignState
    from pyecr.services.dct_service import DctPresentation
    from pyecr.project.dct_archive import DctArchive
    from pyecr.contracts import SimulationResult
    if isinstance(result,AnalysisSnapshot):module='Analysis'
    elif isinstance(result,BatchAnalysisResult):module='Batch'
    elif isinstance(result,GlobalAnalysisResult):module='Global'
    elif isinstance(result,PairDesignState):module='Pair'
    elif isinstance(result,(DctPresentation,DctArchive)):module='DCT'
    elif isinstance(result,SimulationResult):module='Simulation'
    else:raise TypeError('No figure adapter for this result type')
    payload=dict(result=result.to_dict())
    if module in ('Batch','Global'):
        payload['template']=None if template is None else template.to_dict()
        payload['entries']=[dict(record_id=e.record_id,dataset=_dataset_to_dict(e.dataset),temperature=e.temperature,temperature_unit=e.temperature_unit.value,
            analysis_config=None if e.analysis_config is None else e.analysis_config.to_dict(),source_project=e.source_project,
            noise_weight=e.noise_weight,fit_baseline=e.fit_baseline,fit_amplitude=e.fit_amplitude,baseline=e.baseline,amplitude=e.amplitude) for e in entries]
    if module=='DCT':payload['interpretations']=interpretations
    return FigureSource(module,type(result).__name__,freeze(payload),scope,label,view_state)

def catalog(source):
    from pyecr.report.figure_sources import describe_source
    return tuple(describe_source(source))
