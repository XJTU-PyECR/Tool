"""Experimental Signed evidence, separate from current jobs and legacy archives.

Reading validates frozen evidence only. No inversion, lambda selection, Fit,
deconvolution, migration or current publication is available from this module.
"""
from dataclasses import dataclass
from pyecr.contracts import ResultState
from pyecr.contracts.common import require,exact_fields,canonical_sha256,strict_json_loads
from pyecr.dct.prior_selection import read_prior_search,PriorSearchReport
from pyecr.dct.prior_diagnostics import diagnose_prior_spectrum


@dataclass(frozen=True,slots=True)
class UnsupportedPriorArchive:
    reason: str
    payload_sha256: str

    @property
    def state(self):return ResultState(status='unsupported',blocking_reasons=(self.reason,))


@dataclass(frozen=True,slots=True)
class SignedPriorArchive:
    report: PriorSearchReport
    provenance: tuple=()
    schema_version: int=1

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version==1,'unsupported prior archive version')
        require(isinstance(self.report,PriorSearchReport),'typed prior report required')
        object.__setattr__(self,'provenance',tuple(tuple(p) for p in self.provenance))
        require(all(len(p)==2 and all(isinstance(v,str) for v in p) for p in self.provenance),'invalid prior provenance')
        require(len({p[0] for p in self.provenance})==len(self.provenance),'duplicate provenance keys')

    @property
    def state(self):
        r=self.report.selected_result
        return r.state if r is not None else ResultState(status='error',blocking_reasons=('No selected converged spectrum: '+self.report.termination,))

    @property
    def diagnostics(self):
        r=self.report.selected_result
        return diagnose_prior_spectrum(r,search_report=self.report) if r else dict(status='FAIL',reason=self.report.termination,physical_reliability='NOT_ESTABLISHED')

    def to_dict(self):
        report=self.report.to_dict()
        return dict(kind='signed_prior_archive',schema_version=1,report=report,report_sha256=canonical_sha256(report),
            versions=dict(algorithm=self.report.config.algorithm_version,solver=self.report.config.solver,
                selection=self.report.policy.version,boundary='cosine-decades/1' if self.report.config.prior_spec.family!='up2012_mass_reference' else 'paper-box-decades/1'),
            diagnostics=self.diagnostics,provenance=[list(p) for p in self.provenance],
            scope='EXPERIMENTAL_NOT_PROMOTED; frozen whole Signed evidence, no automatic H/O, shape normalization or current publication')

    @classmethod
    def from_dict(cls,value):
        require(isinstance(value,dict),'prior archive mapping required')
        if value.get('kind')!='signed_prior_archive' or type(value.get('schema_version')) is not int or value['schema_version']!=1:
            return UnsupportedPriorArchive('Unsupported Signed prior archive kind/version',canonical_sha256(value))
        exact_fields(value,('kind','schema_version','report','report_sha256','versions','diagnostics','provenance','scope'),'prior archive')
        require(isinstance(value['report'],dict),'prior report mapping required')
        config=value['report'].get('config',{})
        require(isinstance(config,dict) and isinstance(config.get('prior_spec'),dict),'prior config/spec mapping required')
        if config.get('algorithm_version')!='signed-priors/1' or config.get('prior_spec',{}).get('version')!='signed-priors/1':
            return UnsupportedPriorArchive('Unsupported Signed prior algorithm version',canonical_sha256(value))
        require(canonical_sha256(value['report'])==value['report_sha256'],'prior report hash conflict')
        report=read_prior_search(value['report']);obj=cls(report,tuple(tuple(p) for p in value['provenance']))
        require(canonical_sha256(obj.to_dict())==canonical_sha256(value),'prior archived diagnostics/versions/source conflict')
        return obj

    @classmethod
    def from_json(cls,payload):return cls.from_dict(strict_json_loads(payload))

    def validated(self):
        obj=self.from_dict(self.to_dict());require(isinstance(obj,SignedPriorArchive),'prior archive became unsupported');return obj
