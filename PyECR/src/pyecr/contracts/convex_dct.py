"""Frozen GUI evidence for the existing experimental Signed convex engine.

This is a separate analysis kind, never a relabelled SignedDctAnalysis/GCV.
Reads validate arrays and versioned geometry only; no inverse or lambda search.
"""
from dataclasses import dataclass,asdict,field,replace
import json
from math import isfinite,log10
import numpy as np
from pyecr.contracts.common import require,exact_fields,canonical_sha256,strict_json_loads
from pyecr.contracts import DctSourceSnapshot,SignedDctConfig,SignedDctDiagnosticConfig,ResultState


@dataclass(frozen=True,slots=True)
class ConvexDctOptions:
    mode:str='Balanced'
    selection:str='manual'
    noise_sd_physical:float=0.
    noise_evidence:str=''
    ou_seconds:float=0.
    version:str='convex-dct-gui/1'

    def __post_init__(self):
        require(self.version=='convex-dct-gui/1','unsupported convex GUI version')
        require(self.mode in {'Balanced','Suppression'},'unknown Whole Signed mode')
        require(self.selection in {'manual','discrepancy'},'convex modes support Manual or known-noise auto, not GCV')
        for name in ('noise_sd_physical','ou_seconds'):
            require(type(getattr(self,name)) in (int,float) and isfinite(getattr(self,name)) and getattr(self,name)>=0,'invalid '+name)
        require(isinstance(self.noise_evidence,str),'noise evidence text required')
        if self.selection=='discrepancy':require(self.noise_sd_physical>0 and bool(self.noise_evidence.strip()),'Known-noise auto requires a positive physical noise SD and its independent evidence; otherwise choose Manual')
        if self.ou_seconds:require(bool(self.noise_evidence.strip()),'OU correlation time requires independent noise evidence')

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,v):
        exact_fields(v,tuple(cls.__dataclass_fields__),'convex GUI options');return cls(**v)

    def prior_spec(self):
        from pyecr.dct.signed_convex import ConvexSpec
        return ConvexSpec(knot_step_decades=.25,mass_ratio=1. if self.mode=='Balanced' else 16.,ou_correlation_seconds=self.ou_seconds)


@dataclass(frozen=True,slots=True)
class ConvexDctResult:
    source:DctSourceSnapshot
    config:SignedDctConfig
    payload_json:str
    @property
    def payload(self):return strict_json_loads(self.payload_json)
    @property
    def selected_lambda(self):return self.payload['lambda_value']
    @property
    def state(self):return ResultState(status='warning' if self.payload['status']=='CONVERGED' else 'error',warnings=('EXPERIMENTAL_CONVEX_MODE: conditional spectrum, not physical reliability',) if self.payload['status']=='CONVERGED' else (),blocking_reasons=() if self.payload['status']=='CONVERGED' else (self.payload['status'],))
    def _array(self,name):return tuple(self.payload.get(name) or ())
    tau_s=property(lambda self:self._array('tau_s') if self.state.ready else ())
    chi=property(lambda self:self._array('chi'))
    quadrature_weights=property(lambda self:self._array('quadrature_weights') if self.state.ready else ())
    quadrature_mass=property(lambda self:self._array('quadrature_mass'))
    reconstruction=property(lambda self:self._array('reconstruction'))
    residuals=property(lambda self:self._array('residuals'))
    log10_tau=property(lambda self:tuple(log10(t) for t in self.tau_s))
    physical_unit=property(lambda self:self.source.normalization.spec.signal_unit)
    physical_chi=property(lambda self:tuple(self.source.normalization.scale_value*v for v in self.chi))
    physical_mass=property(lambda self:tuple(self.source.normalization.scale_value*v for v in self.quadrature_mass))
    physical_reconstruction=property(lambda self:tuple(self.source.normalization.reference_value+self.source.normalization.scale_value*v for v in self.reconstruction))
    physical_residuals=property(lambda self:tuple(self.source.normalization.scale_value*v for v in self.residuals))
    def to_dict(self):return self.payload


@dataclass(frozen=True,slots=True)
class ConvexDctCandidate:
    result:ConvexDctResult
    score:float|None
    failure:str|None
    # These fields remain unavailable, never repurpose nonlinear scores as GCV.
    gcv_score:None=None
    effective_df:None=None
    raw_effective_df:None=None


def validate_manual(report):
    """Independent forward/objective audit; no fabricated noise for Manual."""
    from pyecr.dct.signed_convex import whiten
    require(report['version']=='convex-gui-manual/1','unsupported convex Manual report')
    source=DctSourceSnapshot.from_dict(report['source']);source.require_usable();cfg=SignedDctConfig.from_dict(report['config'])
    require(len(report['trials'])==1,'Manual requires one frozen lambda')
    r=report['trials'][0];lam=cfg.grid.regularization_lambda
    require(r['source_sha256']==source.content_sha256 and r['lambda_value']==lam and r['spec']==report['spec'],'Manual source/config/spec conflict')
    require(r['df'] is None and r['gcv'] is None and r['fallback'] is None,'Manual convex evidence cannot claim GCV or fallback')
    if r['status']!='CONVERGED':
        require(r['status'].startswith('FAIL_') and r['chi'] is None and report['selected_lambda'] is None,'Manual failure has a main spectrum');return
    tau=np.asarray(r['tau_s']);q=np.asarray(r['quadrature_weights']);chi=np.asarray(r['chi']);mass=np.asarray(r['quadrature_mass']);grid=cfg.grid
    require(tau.shape==q.shape==chi.shape==mass.shape==(grid.n_tau,) and np.all(np.isfinite(chi)),'Manual spectrum shape/nonfinite')
    x=np.linspace(np.log10(grid.tau_min_s),np.log10(grid.tau_max_s),grid.n_tau);h=x[1]-x[0];expected=np.full(len(x),h);expected[[0,-1]]/=2
    require(np.allclose(np.log10(tau),x,rtol=0,atol=2e-13) and np.allclose(q,expected,rtol=1e-12,atol=1e-14),'Manual grid/quadrature conflict')
    require(np.allclose(mass,q*chi,rtol=1e-12,atol=1e-13),'Manual density/mass conflict')
    pred=np.empty(len(source.signal));t=np.asarray(source.time_s)
    for i in range(0,len(t),4096):pred[i:i+4096]=np.exp(-t[i:i+4096,None]/tau)@mass
    residual=np.asarray(source.signal)-pred
    require(np.allclose(pred,r['reconstruction'],rtol=1e-12,atol=1e-12) and np.allclose(residual,r['residuals'],rtol=1e-12,atol=1e-12),'Manual reconstruction conflict')
    w=np.sqrt(np.asarray(cfg.observation_weights or np.ones(len(t))))
    white=whiten(w*residual,t,report['spec']['ou_correlation_seconds']);sse=float(white@white)
    require(np.isclose(sse,r['weighted_sse'],rtol=2e-9,atol=1e-11),'Manual weighted SSE conflict')
    alpha=report['spec']['mass_ratio']*np.sqrt(lam)
    values=[sse,lam*np.sum((np.diff(chi,2)*np.sqrt(h)/h**2)**2),alpha*np.abs(mass).sum()]
    require(report['spec']['edge_beta']==0,'GUI modes use the frozen open-edge prior')
    for name,v in zip(('weighted_sse','curvature_penalty','absolute_mass_penalty'),values):require(np.isclose(v,r[name],rtol=2e-9,atol=1e-11),'Manual objective component conflict')
    require(np.isclose(sum(values),r['objective'],rtol=2e-9,atol=1e-11),'Manual objective conflict')
    require(report['selected_lambda']==lam and r['iterations']<=report['spec']['max_iterations'] and r['kkt']<=report['spec']['kkt_tolerance'],'Manual selection/convergence conflict')


@dataclass(frozen=True,slots=True)
class ConvexDctAnalysis:
    options:ConvexDctOptions
    diagnostic_config:SignedDctDiagnosticConfig
    report_json:str
    source:object=field(init=False,repr=False)
    config:object=field(init=False,repr=False)
    selected_result:object=field(init=False,repr=False)
    candidates:tuple=field(init=False,repr=False)
    peak_diagnostics:tuple=field(init=False,repr=False)
    flags:tuple=field(init=False)
    resolution_status:str=field(init=False)
    state:object=field(init=False)

    def __post_init__(self):
        require(isinstance(self.options,ConvexDctOptions) and isinstance(self.diagnostic_config,SignedDctDiagnosticConfig),'typed convex options/diagnostics required')
        r=strict_json_loads(self.report_json)
        if self.options.selection=='discrepancy':
            from pyecr.report.convex_bundle import validate_report
            validate_report(r)
        else:validate_manual(r)
        require(r['spec']==self.options.prior_spec().to_dict(),'frozen mode/prior mismatch')
        source=DctSourceSnapshot.from_dict(r['source']);cfg=SignedDctConfig.from_dict(r['config'])
        require(source.normalization is not None and source.component_key is None,'convex GUI requires Whole terminal-referenced observations')
        require(cfg.schema_version==2 and cfg.solver=='reused_svd' and cfg.grid.lambda_mode=='manual','convex GUI requires explicit static v2 Manual base config')
        if self.options.selection=='discrepancy':
            require(r['noise_evidence']==self.options.noise_evidence and all(v==self.options.noise_sd_physical for v in r['noise_sd_physical']),'noise declaration conflict')
        candidates=[];selected=None
        for trial in r['trials']:
            tc=replace(cfg,grid=replace(cfg.grid,regularization_lambda=trial['lambda_value']))
            result=ConvexDctResult(source,tc,json.dumps(trial,ensure_ascii=False,allow_nan=False))
            candidates.append(ConvexDctCandidate(result,trial.get('selection_score'),None if trial['status']=='CONVERGED' else trial['status']))
            if trial['lambda_value']==r['selected_lambda']:selected=result
        require(selected is None or selected.state.ready,'selected convex trial is not converged')
        from pyecr.dct.signed_peaks import extract_signed_peaks
        peaks,flags,resolution=extract_signed_peaks(selected.tau_s,selected.chi,self.diagnostic_config) if selected else ((),(),'not_evaluated')
        flags=tuple(dict.fromkeys((*flags,'EXPERIMENTAL_CONVEX_MODE','SENSITIVITY_NOT_RUN')))
        for name,v in dict(source=source,config=cfg,selected_result=selected,candidates=tuple(candidates),peak_diagnostics=tuple(peaks),flags=flags,resolution_status=resolution,
            state=ResultState(status='warning' if selected else 'error',warnings=('Experimental mode: true weak/slow negative contributions may be lost; no calibrated CI or automatic H/O.',) if selected else (),blocking_reasons=() if selected else ('No converged convex candidate',))).items():object.__setattr__(self,name,v)

    report=property(lambda self:strict_json_loads(self.report_json))
    comparisons=property(lambda self:())
    interpretation=property(lambda self:('Conditional on frozen source, scale, mode and declared noise. Stability comparisons were not run.',))
    score_name=property(lambda self:'Discrepancy distance' if self.options.selection=='discrepancy' else 'Manual (no score)')
    content_sha256=property(lambda self:canonical_sha256(self.to_dict()))
    def to_dict(self):
        r=self.report
        return dict(kind='convex_dct_gui_analysis',schema_version=1,options=self.options.to_dict(),diagnostic_config=self.diagnostic_config.to_dict(),
            report=r,report_sha256=canonical_sha256(r),peaks=[p.to_dict() for p in self.peak_diagnostics],flags=list(self.flags),resolution_status=self.resolution_status)
    def to_json(self):return json.dumps(self.to_dict(),ensure_ascii=False,allow_nan=False)
    @classmethod
    def from_dict(cls,v):
        exact_fields(v,('kind','schema_version','options','diagnostic_config','report','report_sha256','peaks','flags','resolution_status'),'convex GUI analysis')
        require(v['kind']=='convex_dct_gui_analysis' and type(v['schema_version']) is int and v['schema_version']==1,'unsupported convex GUI analysis version')
        require(canonical_sha256(v['report'])==v['report_sha256'],'convex GUI report hash conflict')
        obj=cls(ConvexDctOptions.from_dict(v['options']),SignedDctDiagnosticConfig.from_dict(v['diagnostic_config']),json.dumps(v['report'],ensure_ascii=False,allow_nan=False))
        require(canonical_sha256(obj.to_dict())==canonical_sha256(v),'convex GUI frozen geometry/diagnostic conflict');return obj
