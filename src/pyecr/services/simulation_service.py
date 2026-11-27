"""Given-parameter production simulation and optional local observation noise."""
from collections.abc import Mapping
from concurrent.futures import CancelledError
from threading import Lock
from importlib.metadata import PackageNotFoundError, version
from math import isfinite, isnan, log10
import platform
import warnings

import numpy as np
import scipy

from pyecr.contracts import SimulationRequest, SimulationResult, ResultState
from pyecr.contracts.simulation import SIMULATION_ADAPTER_VERSION, SIMULATION_INTERPRETATION, SIMULATION_POINT_BUDGET
from pyecr.contracts.simulation_observation import observation_ledger
from pyecr.model_display import MODEL_DISPLAY, BY_ID, VERSION as DISPLAY_VERSION
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.boundary.composition import BoundaryComposedSingleEcrModel
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.dual_surface_k import DualSurfaceKModel
from pyecr.models.surface_k import SurfaceKModel, MODEL_VERSION, EXPOSURE_RULE
from pyecr.models.component_cache import ComponentCacheScope
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.models.single import SingleEcrContext, SingleEcrModel
from pyecr.models.root_cache import RootCacheScope, ROOT_SOLVER_POLICY

__all__ = ['SimulationService', 'simulation_service_support']

SUPPORTED_MODELS = frozenset(ModelID(item.model_id) for item in MODEL_DISPLAY)
# Production warning capture uses process-global Python warning filters. Keep
# forward calls serialized; independent jobs may queue, edit and cancel safely.
_FORWARD_LOCK = Lock()


def simulation_service_support():
    """Actual production adapters; boundary support remains model-specific."""
    return dict(version='simulation-service-support/3', adapter_version=SIMULATION_ADAPTER_VERSION,
        models=[item.model_id for item in MODEL_DISPLAY], display_version=DISPLAY_VERSION,
        model_matrix=[dict(model_id=item.model_id, menu_number=item.number, menu_alias=item.title,
            display_label=item.label, surface_modes=list(DEFAULT_MODEL_REGISTRY.get(item.model_id).supported_surface_modes),
            boundary_modes=list(DEFAULT_MODEL_REGISTRY.get(item.model_id).supported_boundary_modes)) for item in MODEL_DISPLAY],
        # Preserve the P11 key as the intersection, never the union of modes.
        boundary_modes=['ideal_step'], boundary_modes_scope='common_to_all_listed_models',
        noise_sd=0., noise_modes=['none','gaussian_iid'], rng_algorithm='PCG64', scientific_point_budget=SIMULATION_POINT_BUDGET,
        geometry='complete_dimensions_cm',
        exposed_faces='both_opposite_faces_per_active_axis', signal='baseline_plus_signed_completion',
        result_schema_version=4, statistical_reliability='not_evaluated',
        not_implemented=['dual_nonideal_boundaries', 'k_only_nonideal_boundaries', 'Analysis_apply'])


def _plain(value, path, nonfinite_paths):
    """Retain nonfinite diagnostics as explicit tagged values, never fake zeros."""
    if isinstance(value, Mapping):
        return {k:_plain(v, path+'.'+k, nonfinite_paths) for k,v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), path, nonfinite_paths)
    if isinstance(value, np.generic):
        return _plain(value.item(), path, nonfinite_paths)
    if isinstance(value, (list, tuple)):
        return [_plain(v, f'{path}[{i}]', nonfinite_paths) for i,v in enumerate(value)]
    if isinstance(value, float) and not isfinite(value):
        nonfinite_paths.append(path)
        return {'nonfinite': 'nan' if isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity'}
    return value


def _software_versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pyecr='PyECR')


class SimulationService:
    """Stateless service: run one immutable request through the production factory.

    Valid-but-not-implemented requests return Unsupported with no arrays.
    Malformed requests are rejected by SimulationRequest before execution.
    Numerical failures return Error, never an old curve or a synthetic FitResult.
    """

    def run(self, request: SimulationRequest, *, cancel_token=None, progress_callback=None) -> SimulationResult:
        if not isinstance(request, SimulationRequest):
            raise TypeError('SimulationRequest required; no dataset/AnalysisConfig/FitResult adapter is implicit')
        check = cancel_token.raise_if_cancelled if cancel_token is not None else lambda: None
        progress = progress_callback or (lambda fraction, message: None)
        # Historical v1 payloads remain readable; execution uses today's bounded
        # allocation policy rather than clipping an old scientific grid.
        if len(request.time_s)>SIMULATION_POINT_BUDGET:
            check()
            return SimulationResult(schema_version=4 if request.schema_version==2 or request.noise_sd>0 else 3,
                request=request,state=ResultState(status='unsupported',blocking_reasons=(
                    f'SCIENTIFIC_POINT_BUDGET: {len(request.time_s)} exceeds {SIMULATION_POINT_BUDGET}; no points truncated',)))
        check(); progress(0., 'Waiting for production forward slot; phase progress, not kernel iteration percent')
        while not _FORWARD_LOCK.acquire(timeout=.05):check()
        try:
            check(); progress(.1, 'Production forward running on the complete scientific grid; internal progress unavailable')
            result = self._run(request, check, progress)
            check()
            if result.state.ready:progress(1., 'Complete simulated observations validated; awaiting publication')
            check()
            return result
        finally:
            _FORWARD_LOCK.release()

    def _run(self, request, check, report):
        result_version = 4 if request.schema_version >= 2 or request.noise_sd > 0 else 3
        messages, codes, captured = [], [], []
        def warn(code, message):
            messages.append(f'{code}: {message}')
            if code not in codes:codes.append(code)

        try:
            params = request.given_parameters
            times = np.asarray(request.time_s)
            nonfinite_paths = []
            components, parts = {}, {}
            evaluated = dict(params)

            def series_evidence(kernel, prefix):
                unconverged = [name for name, axis in zip('xyz', kernel.axis_results)
                    if axis is not None and not axis.diagnostics['converged']]
                if unconverged:
                    warn('SERIES_TRUNCATION_UNRESOLVED', prefix+': production term budget unresolved on '+', '.join(unconverged)+'; no coefficient normalization or clipping')
                if any(a is not None and a.diagnostics.get('asymptotic_roots_used') for a in kernel.axis_results):
                    warn('LARGE_BI_ROOT_APPROXIMATION', prefix+': production large-Bi asymptotic roots used')
                axes = [None if a is None else dict(remaining_fraction=a.F.tolist(), n_terms_used=a.n_terms_used,
                    coefficient_sum=a.coefficient_sum, diagnostics=_plain(a.diagnostics, prefix+f'axis_results[{i}].diagnostics', nonfinite_paths))
                    for i,a in enumerate(kernel.axis_results)]
                return axes, _plain(kernel.diagnostics, prefix+'kernel_diagnostics', nonfinite_paths)

            def component(key, role, sign, amplitude, progress, ideal, kinetics, kernel=None, rates=None):
                progress = np.asarray(progress)
                # Ideal diffusion F comes directly from the production kernel.
                remaining = np.asarray(kernel.F) if kernel is not None and request.boundary.mode == 'ideal_step' else 1.-progress
                prefix = '' if len(parts)==1 else 'component_evidence.'+key+'.'
                axes, diagnostics = series_evidence(kernel, prefix) if kernel is not None else (None, None)
                components[key] = dict(role=role, sign=sign, amplitude=amplitude, kinetics=kinetics,
                    remaining_fraction=remaining.tolist(), completion_fraction=progress.tolist(),
                    ideal_completion_fraction=np.asarray(ideal).tolist(), axis_results=axes,
                    kernel_diagnostics=diagnostics, surface_rates=rates, speed_label=None)
                if np.any((remaining < 0.) | (remaining > 1.)):
                    warn('REMAINING_FRACTION_OUTSIDE_RANGE', key+': production response retained without clipping')
                if amplitude == 0:
                    warn('ZERO_AMPLITUDE', key+': flat given component carries no kinetic information')

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', RuntimeWarning)
                # Both task-owned caches are isolated from surrounding Analysis.
                with RootCacheScope(enabled=False, root_mode='legacy_full_roots', cancel_check=check), ComponentCacheScope(enabled=False):
                    model = ModelFactory().create(request.model) if request.boundary.mode == 'ideal_step' else ModelFactory().create(request.model, boundary=request.boundary)
                    boundary_diagnostics = dict(type='ideal_step', composition_applied=False,
                        extrapolation_used=False, time_origin_s=0.)
                    if isinstance(model, (SingleEcrModel, BoundaryComposedSingleEcrModel)):
                        theta = {'log10_D':log10(params['D_cm2_s'])}
                        if request.model.model_id is ModelID.M1_SINGLE_DK:theta['log10_k']=log10(params['k_cm_s'])
                        context = SingleEcrContext(amplitude=params['amplitude'], baseline=params['baseline'],
                            term_tol=request.solver_config.relative_tolerance, max_terms=request.solver_config.max_terms)
                        prediction = model.evaluate(times, theta, context)
                        material = prediction.material_prediction if isinstance(model, BoundaryComposedSingleEcrModel) else prediction
                        if material is None:raise ValueError('production boundary prediction omitted its material evidence')
                        progress = prediction.normalized_response if isinstance(model, BoundaryComposedSingleEcrModel) else 1.-material.F
                        ideal = prediction.ideal_response if isinstance(model, BoundaryComposedSingleEcrModel) else progress
                        parts = {'component_1':params['amplitude']*progress}
                        evaluated['D_cm2_s'] = material.D_cm2_s
                        if material.k_cm_s is not None:evaluated['k_cm_s']=material.k_cm_s
                        component('component_1', 'single', 1., params['amplitude'], progress, ideal,
                            {k:v for k,v in evaluated.items() if k in {'D_cm2_s','k_cm_s'}}, material.kernel_result)
                        clean = prediction.y
                        if isinstance(model, BoundaryComposedSingleEcrModel):
                            boundary_diagnostics = _plain(prediction.diagnostics, 'boundary_diagnostics', nonfinite_paths)
                            boundary_diagnostics['time_origin_s'] = request.boundary.measured_time_s[0] if request.boundary.mode == 'measured_history' else 0.
                            boundary_diagnostics['extrapolation_used'] = False
                            summary = boundary_diagnostics['material_kernel_evaluation_summary']
                            if not summary['all_axes_converged']:
                                warn('SERIES_TRUNCATION_UNRESOLVED', 'boundary callbacks include '+str(summary['unconverged_evaluations'])+
                                    ' unresolved production series evaluations; requested-grid evidence alone is insufficient')
                            if summary['asymptotic_roots_used']:
                                warn('LARGE_BI_ROOT_APPROXIMATION', 'boundary callbacks used recorded production asymptotic roots')
                            if request.boundary.mode == 'exponential_flush' and boundary_diagnostics['severity'] != 'no_correction_needed':
                                warn('BOUNDARY_SEVERITY', boundary_diagnostics['severity']+'; this is boundary screening, not statistical reliability')
                            for message in boundary_diagnostics['core'].get('warnings', ()):
                                warn('BOUNDARY_NUMERICAL_WARNING', str(message))
                            if request.boundary.mode == 'measured_history' and request.boundary.measured_values[-1] != request.boundary.measured_final_value:
                                warn('MEASURED_FINAL_NOT_REACHED', 'declared final gas value differs from last history value; asymptote is conditional, not an extrapolated observation')
                    elif isinstance(model, DualHOModel):
                        prediction = model.evaluate(times, **params, kernel_options=dict(
                            term_tol=request.solver_config.relative_tolerance, max_terms=request.solver_config.max_terms))
                        clean = prediction.total
                        parts = dict(component_H=prediction.component_H, component_O=prediction.component_O)
                        for slot in ('H','O'):
                            kinetics = {k:v for k,v in params.items() if k in {f'D_{slot}_cm2_s',f'k_{slot}_cm_s'}}
                            fraction = getattr(prediction, 'F_'+slot)
                            component('component_'+slot, 'parent_model_role_'+slot, getattr(prediction,'sign_'+slot),
                                params['A_'+slot], 1.-fraction, 1.-fraction, kinetics, getattr(prediction, 'kernel_'+slot+'_result'))
                    elif isinstance(model, (SurfaceKModel, DualSurfaceKModel)):
                        if isinstance(model, SurfaceKModel):
                            clean = model.predict(times, params)
                            progress, _ = model.kernel.evaluate(times, params['k_cm_s'])
                            parts = {'component_1':params['amplitude']*progress}
                            slots = [('component_1','single',1.,params['amplitude'],'k_cm_s')]
                        else:
                            clean, contributions = model.evaluate(times, params)
                            parts = dict(zip(('component_1','component_2'), contributions))
                            slots = [(f'component_{i}',f'effective_input_mode_{i}',model.signs[i-1],params[f'A_{i}'],f'k_{i}_cm_s') for i in (1,2)]
                        for key,role,sign,amplitude,kname in slots:
                            progress, _ = model.kernel.evaluate(times, params[kname])
                            rate, tau = model.kernel.rates(params[kname])
                            component(key, role, sign, amplitude, progress, progress, {kname:params[kname]}, rates=dict(
                                geometry_factor_cm_inv=model.kernel.g_cm_inv, rate_s_inv=rate, tau_s=tau,
                                production_kernel_version=MODEL_VERSION, exposure_rule=EXPOSURE_RULE))
                        if isinstance(model, DualSurfaceKModel):
                            r1,r2=(components[k]['surface_rates']['rate_s_inv'] for k in ('component_1','component_2'))
                            for i,key in enumerate(('component_1','component_2')):
                                components[key]['speed_label'] = 'equal_rate' if r1==r2 else 'effective_fast' if (r1>r2)==(i==0) else 'effective_slow'
                    else:
                        raise TypeError('factory returned an unsupported production model type')
            for w in captured:
                warn('PRODUCTION_NUMERICAL_WARNING', f'{w.category.__name__}: {w.message}')
            clean = np.asarray(clean)
            # `clean` above is always the corresponding production prediction.
            # Dual/K interfaces do not expose a .y result object.
            if clean.shape != (len(times),) or not np.all(np.isfinite(clean)) or any(
                    np.asarray(c).shape != (len(times),) or not np.all(np.isfinite(c)) for c in parts.values()):
                raise ValueError('production prediction is nonfinite or omits requested observation times')
            if np.all(clean == clean[0]) and any(np.any(c != c[0]) for c in parts.values()):
                amplitudes = [c['sign']*c['amplitude'] for c in components.values()]
                if sum(amplitudes)==0 and len(parts)==2 and all(np.array_equal(c, -next(iter(parts.values()))) for c in list(parts.values())[1:]):
                    warn('EXACT_COMPONENT_CANCELLATION', 'nonzero signed contributions cancel on the full requested grid')
                else:
                    warn('SIGNAL_CHANGE_BELOW_FLOAT_RESOLUTION', 'baseline addition erased variation in float64 output; signed contributions retained')
            expected_suffix = '.diagnostics.truncation_error_by_time[0]'
            unexpected_nonfinite = [p for p in nonfinite_paths if not (times[0]==0 and p.endswith(expected_suffix))]
            if unexpected_nonfinite:
                warn('NONFINITE_PRODUCTION_DIAGNOSTIC', 'tagged without replacement: '+', '.join(unexpected_nonfinite))
            single = components.get('component_1') if len(components)==1 else None
            display = BY_ID[request.model.model_id.value]
            baseline = params['baseline'] if 'baseline' in params else params['y0']
            has_series = any(c['axis_results'] is not None for c in components.values())
            evidence = dict(adapter_version=SIMULATION_ADAPTER_VERSION, request_sha256=request.content_sha256,
                source_kind='simulation_clean', time_unit='s', signal_unit=request.signal_unit, length_unit='cm',
                parameter_units={p.name:request.signal_unit if p.unit=='signal' else p.unit
                    for p in ModelFactory().registry.get(request.model.model_id).parameters},
                geometry_convention='complete_dimensions_cm', active_face_convention='both_opposite_faces_per_active_axis',
                signal_convention='baseline_plus_signed_completion', term_tol=request.solver_config.relative_tolerance,
                absolute_tolerance_usage='recorded_not_used_by_single_series_kernel' if has_series else 'not_used_by_analytic_surface_kernel',
                root_mode='legacy_full_roots' if has_series else 'not_applicable', root_solver_policy=ROOT_SOLVER_POLICY if has_series else 'not_applicable', software_versions=_software_versions(),
                remaining_fraction=single['remaining_fraction'] if single else None,
                completion_fraction=single['completion_fraction'] if single else None,
                axis_results=single['axis_results'] if single else None, kernel_diagnostics=single['kernel_diagnostics'] if single else None,
                nonfinite_diagnostic_paths=nonfinite_paths,
                t_zero_rule='production_analytic_initial_condition; series_bound_not_applicable_at_zero',
                evaluated_parameters=evaluated, numerical_warning_codes=codes,
                interpretation=SIMULATION_INTERPRETATION,
                model_display=dict(version=DISPLAY_VERSION, model_id=display.model_id, menu_number=display.number,
                    menu_alias=display.title, display_label=display.label), boundary_diagnostics=boundary_diagnostics,
                component_evidence=components,
                given_diffusion_background={k:v for k,v in params.items() if k.startswith('D_')}
                    if request.model.model_id in {ModelID.M1_SINGLE_DK,ModelID.M3_DUAL_HO_DK} else {},
                limit_signals=dict(initial_time_s=boundary_diagnostics['time_origin_s'], initial_signal=baseline,
                    asymptotic_signal=baseline+sum(c['sign']*c['amplitude'] for c in components.values()),
                    asymptotic_basis='conditional_declared_final_gas' if request.boundary.mode=='measured_history' else 'given_parameters_ideal_or_flush_limit'))
            check(); report(.7, 'Clean full-grid forward returned; preparing one total observation noise vector')
            check()
            if request.noise_sd > 0:
                # No global seed/state, job-id hashing, component noise or
                # implicit fresh entropy. Sequence is an explicit spawn key.
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                    request.seed, spawn_key=(request.sequence_number,))))
                with np.errstate(over='ignore', invalid='ignore'):
                    noise = rng.normal(0., request.noise_sd, len(clean))
            else:
                noise = np.zeros(len(clean), dtype=float)
            with np.errstate(over='ignore', invalid='ignore'):
                observed = clean + noise
            if not (np.all(np.isfinite(noise)) and np.all(np.isfinite(observed))):
                raise ValueError('OBSERVATION_NOISE_NONFINITE: noise or clean+noise overflow; no clipping or partial publication')
            check(); report(.85, 'Validating full clean/noise/observed arrays and source identities')
            return SimulationResult(schema_version=result_version, request=request,
                state=ResultState(status='warning', warnings=tuple(messages)) if messages else ResultState(),
                signal_clean=tuple(clean), signal_observed=tuple(observed), noise=tuple(noise),
                contributions={k:tuple(v) for k,v in parts.items()}, numerical_evidence=evidence,
                observation_evidence=observation_ledger(request,np.__version__) if result_version == 4 else None)
        except CancelledError:
            raise
        except Exception as exc:
            for w in captured:
                message = f'{w.category.__name__}: {w.message}'
                if f'PRODUCTION_NUMERICAL_WARNING: {message}' not in messages:
                    warn('PRODUCTION_NUMERICAL_WARNING', message)
            return SimulationResult(schema_version=result_version, request=request, state=ResultState(status='error',
                warnings=tuple(messages), blocking_reasons=(f'FORWARD_COMPUTATION_FAILED: {type(exc).__name__}: {exc}',)))
