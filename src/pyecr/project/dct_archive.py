"""P10 portable DCT history. Reading validates evidence, never fits or inverts."""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import numpy as np

from pyecr.contracts import (ComponentDctBundle, ComponentDctConfig, ComponentSourceCapture,
    DctSourceSnapshot, SignedDctAnalysis, SignedDctConfig, SignedDctDiagnosticConfig,
    SignedNormalizationResult, SignedNormalizationSpec, SourceIdentity)
from pyecr.contracts.common import (exact_fields, require, integer, number, text,
    canonical_sha256, strict_json_loads)
from pyecr.preprocessing import PreprocessConfig
from pyecr.report.snapshot import AnalysisSnapshot, _dataset_to_dict, _dataset_from_dict
from pyecr.services.dct_sources import PreparedSignedDctInput
from pyecr.services.signed_dct_service import (DctPhysicalContext, SignedDctRequest,
    SignedDctProgress, SignedDctJob)
from pyecr.services.component_dct_service import ComponentDctRequest, ComponentDctJob


def _version(v, name):
    require(integer(v['schema_version'], name) == 1, f'unknown {name} version')


def _optional(cls, value):
    return None if value is None else cls.from_dict(value)


def _same(a, b, name):
    require(canonical_sha256(a) == canonical_sha256(b), f'{name} content/identity conflict')


def _physical(v):
    exact_fields(v, DctPhysicalContext().to_dict(), 'physical context')
    obj = DctPhysicalContext(**{k: v[k] for k in ('schema_version', 'boundary_mode',
        'model_id', 'config_sha256', 'geometry_cm', 'active_dimensions')})
    _same(obj.to_dict(), v, 'physical context')
    return obj


def _normalization_binding(dataset, recipe, norm, *, parent_identity=None):
    """Integrity replay of the existing pure P02 calculation; no dataset/recipe write."""
    from pyecr.preprocessing import normalize_signed_dct
    from pyecr.preprocessing.signed_normalization import applied_signed_normalization
    identity = SourceIdentity.capture(dataset, recipe)
    if dataset.metadata.get('normalization_mode') == 'signed_terminal':
        expected = applied_signed_normalization(dataset, recipe=recipe)
    else:
        expected = normalize_signed_dct(dataset, replace(norm.spec, identity=identity), recipe=recipe)
    if parent_identity is not None:
        expected = replace(expected, spec=replace(expected.spec, identity=parent_identity), output_identity=None)
    require(norm == expected, 'normalization/source/terminal/scale audit conflict')


def _prepared(v, request):
    if v is None:
        return None
    exact_fields(v, ('schema_version', 'processed', 'recipe', 'raw_sha256', 'mask',
        'normalization', 'source', 'terminal_sources', 'terminal_preparation_failure'), 'prepared input')
    _version(v, 'prepared input')
    obj = PreparedSignedDctInput(DctSourceSnapshot.from_dict(v['processed']),
        PreprocessConfig.from_dict(v['recipe']), v['raw_sha256'],
        SignedNormalizationResult.from_dict(v['normalization']),
        _optional(DctSourceSnapshot, v['source']),
        tuple(DctSourceSnapshot.from_dict(s) for s in v['terminal_sources']),
        v['terminal_preparation_failure'])
    _same(obj.to_dict(), v, 'prepared input')
    _same(obj.recipe.to_dict(), request.recipe.to_dict(), 'prepared recipe')
    ds = request.dataset
    from pyecr.preprocessing.signed_normalization import raw_observations_sha256, cumulative_t0_s
    require(obj.raw_sha256 == raw_observations_sha256(ds), 'prepared raw hash conflict')
    _normalization_binding(ds, request.recipe, obj.normalization)
    expected = DctSourceSnapshot.capture_experiment(ds, request.recipe,
        t0_s=cumulative_t0_s(ds),
        normalization=obj.normalization if ds.metadata.get('normalization_mode') == 'signed_terminal' else None)
    require(obj.processed == expected, 'processed observation/source conflict')
    if obj.source is not None:
        require(obj.source == DctSourceSnapshot.capture_experiment(ds, request.recipe,
            normalization=obj.normalization, t0_s=cumulative_t0_s(ds)), 'normalized source conflict')
    return obj


def _lifecycle(job, payload):
    require(job.status in {'queued', 'running', 'completed', 'warning', 'error', 'unsupported', 'cancelled', 'stale'},
        'unknown DCT job status')
    text(job.job_id, 'job_id')
    require(isinstance(job.cancellation_requested, bool), 'cancellation_requested must be boolean')
    require(all(isinstance(v, str) and v for v in job.stale_reasons), 'invalid stale reasons')
    require(job.status != 'stale' or job.stale_reasons, 'stale history needs reason')
    require(job.failure is None or isinstance(job.failure, str), 'invalid job failure')
    require(job.status not in {'completed', 'warning'} or payload is not None and payload.state.ready,
        'completed job missing ready scientific payload')
    previous = 0.
    for i, p in enumerate(job.progress):
        f, msg = (p.fraction, p.message) if isinstance(p, SignedDctProgress) else p
        require(previous <= number(f, 'progress') <= 1., 'nonmonotonic progress')
        text(msg, 'progress message'); previous = f
        if isinstance(p, SignedDctProgress):
            require(integer(p.sequence, 'sequence') == i+1, 'progress sequence conflict')


def read_whole(entry):
    exact_fields(entry, ('dataset', 'job'), 'whole archive entry')
    v = entry['job']; _version(v, 'whole job')
    ds = _dataset_from_dict(entry['dataset'])
    _same(_dataset_to_dict(ds), entry['dataset'], 'full dataset')
    from pyecr.contracts.convex_dct import ConvexDctOptions,ConvexDctAnalysis
    options=None if 'convex_options' not in v else ConvexDctOptions.from_dict(v['convex_options'])
    req = SignedDctRequest(ds, PreprocessConfig.from_dict(v['recipe']),
        SourceIdentity.from_dict(v['identity']), SignedDctConfig.from_dict(v['config']),
        SignedDctDiagnosticConfig.from_dict(v['diagnostic_config']),
        _optional(SignedNormalizationSpec, v['normalization_spec']), _physical(v['physical_context']),convex_options=options)
    if req.normalization_spec is not None:
        req.identity.assert_matches(req.normalization_spec.identity)
    j = SignedDctJob(v['job_id'], req, integer(v['generation'], 'generation'), v['status'],
        tuple(SignedDctProgress(**p) for p in v['progress']), _prepared(v['prepared'], req),
        _optional(ConvexDctAnalysis if options is not None else SignedDctAnalysis, v['analysis']), tuple(v['stale_reasons']),
        v['cancellation_requested'], v['failure'], v['exception_type'])
    _same(j.to_dict(), v, 'whole job')  # also rejects missing/extra redundant evidence
    if j.analysis is not None:
        require(j.prepared is not None and j.analysis.source == j.prepared.source,
            'analysis/prepared source conflict')
        require(j.analysis.config == req.config and j.analysis.diagnostic_config == req.diagnostic_config,
            'analysis/request configuration conflict')
        if options is not None:require(j.analysis.options==options,'convex analysis/request mode or noise conflict')
    _lifecycle(j, j.analysis)
    return j


def read_component(v):
    _version(v, 'component job')
    r = v['request']; _version(r, 'component request')
    parent = AnalysisSnapshot.from_dict(r['parent_snapshot'])
    req = ComponentDctRequest(parent.dataset, PreprocessConfig.from_dict(r['recipe']),
        ComponentSourceCapture.from_dict(r['capture']), SignedDctConfig.from_dict(r['config']),
        tuple(ComponentDctConfig.from_dict(c) for c in r['component_configs']),
        _optional(SignedNormalizationSpec, r['normalization_spec']),
        SignedDctDiagnosticConfig.from_dict(r['diagnostic_config']), parent)
    from pyecr.preprocessing.signed_normalization import raw_observations_sha256, cumulative_t0_s
    c = req.capture
    expected = DctSourceSnapshot.capture_experiment(req.dataset, req.recipe, t0_s=cumulative_t0_s(req.dataset))
    require(c.parent_source == replace(expected, identity=c.identity),
        'captured parent/full observation conflict')
    require(c.raw_sha256 == raw_observations_sha256(req.dataset), 'component raw hash conflict')
    require(np.allclose(c.fit_total, parent.fit_result.y_fit, atol=2e-12, rtol=2e-12),
        'captured parent Fit observation conflict')
    require(len(req.component_configs) == 2, 'two component configs required')
    if req.normalization_spec is not None:
        SourceIdentity.capture(req.dataset, req.recipe).assert_matches(req.normalization_spec.identity)
    j = ComponentDctJob(v['job_id'], req, v['status'], tuple(tuple(p) for p in v['progress']),
        _optional(ComponentDctBundle, v['bundle']), v['failure'], tuple(v['stale_reasons']),
        v['cancellation_requested'])
    _same(j.to_dict(), v, 'component job')
    b = j.bundle
    if b is not None:
        require(b.schema_version == 2 and b.component_capture == req.capture,
            'component bundle/parent capture conflict')
        _normalization_binding(req.dataset, req.recipe, b.parent_source.normalization, parent_identity=c.identity)
        require(b.overall_analysis.config == req.config and
            b.overall_analysis.diagnostic_config == req.diagnostic_config, 'component whole config conflict')
        for result, config in zip(b.independent_results, req.component_configs, strict=True):
            require(result is None or result.config == config, 'component config conflict')
    _lifecycle(j, b)
    return j


@dataclass(frozen=True, slots=True)
class DctArchive:
    """Full captured jobs, independent of current project dataset and legacy DCT."""
    whole_jobs: tuple[SignedDctJob, ...] = ()
    component_jobs: tuple[ComponentDctJob, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'whole_jobs', tuple(self.whole_jobs))
        object.__setattr__(self, 'component_jobs', tuple(self.component_jobs))
        require(all(isinstance(j, SignedDctJob) for j in self.whole_jobs) and
            all(isinstance(j, ComponentDctJob) for j in self.component_jobs), 'typed DCT jobs required')
        keys = [j.job_id for j in (*self.whole_jobs, *self.component_jobs)]
        require(len(set(keys)) == len(keys), 'duplicate DCT job identity')

    def to_dict(self):
        return dict(kind='dct_archive', schema_version=2 if any(j.request.convex_options is not None for j in self.whole_jobs) else 1,
            whole_jobs=[dict(dataset=_dataset_to_dict(j.request.dataset), job=j.to_dict()) for j in self.whole_jobs],
            component_jobs=[j.to_dict() for j in self.component_jobs])

    @classmethod
    def from_dict(cls, value):
        exact_fields(value, ('kind', 'schema_version', 'whole_jobs', 'component_jobs'), 'DCT archive')
        require(type(value['schema_version']) is int and value['schema_version'] in (1,2),'unknown/unsupported DCT archive version')
        require(value['kind'] == 'dct_archive', 'DCT archive kind required')
        require(isinstance(value['whole_jobs'],list) and isinstance(value['component_jobs'],list) and all(isinstance(j,dict) and isinstance(j.get('job'),dict) for j in value['whole_jobs']),'malformed DCT job arrays')
        has_convex=any('convex_options' in j.get('job',{}) for j in value['whole_jobs'])
        require((value['schema_version']==2)==has_convex,'DCT archive mode/version conflict')
        try:
            return cls(tuple(read_whole(j) for j in value['whole_jobs']),
                tuple(read_component(j) for j in value['component_jobs']))
        except (KeyError, TypeError, IndexError) as exc:
            from pyecr.contracts.common import ContractError
            raise ContractError(f'malformed DCT archive: {exc}') from exc

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(strict_json_loads(value))

    def validated(self):
        return self.from_dict(self.to_dict())

    def restore_history(self, whole_service, component_service):
        """Explicit project load: captured status retained on disk, runtime history stale.

        No tokens are created for saved queued/running jobs; nothing resumes or becomes current.
        """
        archive = self.validated()
        reason = 'Restored historical DCT evidence; explicit Run required for a current result'
        whole_service.restore_history(tuple(replace(j, status='stale',
            stale_reasons=tuple(dict.fromkeys((*j.stale_reasons, reason)))) for j in archive.whole_jobs))
        component_service.restore_history(tuple(replace(j, status='stale',
            stale_reasons=tuple(dict.fromkeys((*j.stale_reasons, reason)))) for j in archive.component_jobs))
