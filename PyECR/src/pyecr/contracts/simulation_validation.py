"""P12 payload identity/algebra checks, with no forward execution or fitting."""
from math import prod, fsum
from .common import require, exact_fields, number, integer, vector, close_vectors, text
from .simulation import (SIMULATION_ADAPTER_VERSION, SIMULATION_INTERPRETATION,
    SIMULATION_EVIDENCE_FIELDS, SIMULATION_COMPONENT_FIELDS)
from pyecr.model_display import BY_ID, VERSION as DISPLAY_VERSION
from pyecr.models.ids import ModelID
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.models.root_cache import ROOT_SOLVER_POLICY
from pyecr.models.surface_k import MODEL_VERSION, EXPOSURE_RULE


def validate_simulation_evidence(result):
    r, e = result.request, result.numerical_evidence
    exact_fields(e, SIMULATION_EVIDENCE_FIELDS, 'simulation v3 evidence')
    require(result.schema_version == 4 or r.noise_sd == 0, 'v3 simulation adapter is noise-free')
    require(e['adapter_version'] == SIMULATION_ADAPTER_VERSION and e['request_sha256'] == r.content_sha256,
        'simulation adapter/request identity conflict')
    require(e['source_kind']=='simulation_clean' and e['time_unit']=='s' and e['length_unit']=='cm'
        and e['signal_unit']==r.signal_unit, 'simulation source/units conflict')
    desc = DEFAULT_MODEL_REGISTRY.get(r.model.model_id)
    require(e['parameter_units']=={p.name:r.signal_unit if p.unit=='signal' else p.unit for p in desc.parameters},
        'given-parameter units conflict')
    require(e['geometry_convention']=='complete_dimensions_cm'
        and e['active_face_convention']=='both_opposite_faces_per_active_axis'
        and e['signal_convention']=='baseline_plus_signed_completion', 'geometry/signal convention conflict')
    require(e['term_tol']==r.solver_config.relative_tolerance, 'series tolerance conflict')
    require(e['interpretation']==SIMULATION_INTERPRETATION, 'missing forward-only interpretation')
    require(e['t_zero_rule']=='production_analytic_initial_condition; series_bound_not_applicable_at_zero', 't=0 rule conflict')
    item = BY_ID[r.model.model_id.value]
    require(e['model_display']==dict(version=DISPLAY_VERSION, model_id=item.model_id, menu_number=item.number,
        menu_alias=item.title, display_label=item.label), 'scientific model/menu alias conflict')
    exact_fields(e['evaluated_parameters'], r.given_parameters, 'evaluated physical parameters')
    for name,value in e['evaluated_parameters'].items():
        close_vectors((number(value,name),), (r.given_parameters[name],), 'evaluated given parameter '+name, atol=0, rtol=2e-12)
    exact_fields(e['software_versions'], ('python','numpy','scipy','pyecr'), 'software versions')
    for key,value in e['software_versions'].items():text(value,key)
    codes = e['numerical_warning_codes']
    require(isinstance(codes,tuple) and len(set(codes))==len(codes), 'invalid numerical warning codes')
    for code in codes:
        text(code,'warning code')
        require(any(w.startswith(code+':') for w in result.state.warnings), 'numerical warning was hidden')
    # A history is driven from its recorded start, never silently shifted to the
    # first requested observation. The original request carries every gas row.
    bd = e['boundary_diagnostics']
    origin = r.boundary.measured_time_s[0] if r.boundary.mode=='measured_history' else 0.
    require(bd['type']==r.boundary.mode and bd['extrapolation_used'] is False
        and bd['time_origin_s']==origin and bd['composition_applied']==(r.boundary.mode!='ideal_step'),
        'boundary identity/time origin/extrapolation conflict')
    if r.boundary.mode=='exponential_flush':
        require(bd['flush_time_s']==r.boundary.flush_time_s, 'flush residence time conflict')
        require(bd['severity'] in {'no_correction_needed','correction_recommended','severe_limitation'}, 'boundary severity missing')
        if bd['severity']!='no_correction_needed':require('BOUNDARY_SEVERITY' in codes,'boundary caution hidden')
    elif r.boundary.mode=='measured_history':
        core=bd['core']
        require(bd['signal_type']==r.boundary.measured_signal_type and core['extrapolation_used'] is False
            and core['smoothing_applied'] is False and core['clipping_applied'] is False and core['input_was_fit'] is False,
            'measured boundary provenance conflict')
        require(core['max_internal_points']==r.boundary.max_internal_points
            and integer(core['n_internal_points'],'n_internal_points',minimum=1)<=r.boundary.max_internal_points,
            'convolution point budget conflict')
        require(core['automatic_resampling']==(r.boundary.resample_time_step_s is None), 'convolution sampling declaration conflict')
        if r.boundary.measured_values[-1]!=r.boundary.measured_final_value:
            require('MEASURED_FINAL_NOT_REACHED' in codes, 'conditional measured asymptote caution hidden')
    if r.boundary.mode!='ideal_step':
        summary=bd['material_kernel_evaluation_summary']
        exact_fields(summary,('n_evaluations','n_points','min_time_s','max_time_s','all_axes_converged',
            'unconverged_evaluations','asymptotic_roots_used','max_terms_used_by_axis','scope'),'boundary kernel evaluation summary')
        count=integer(summary['n_evaluations'],'boundary evaluations',minimum=1)
        require(integer(summary['n_points'],'boundary evaluated points',minimum=1)>=len(r.time_s),'boundary summary omits requested points')
        unresolved=integer(summary['unconverged_evaluations'],'unconverged evaluations')
        require(unresolved<=count and summary['all_axes_converged']==(unresolved==0)
            and isinstance(summary['all_axes_converged'],bool) and isinstance(summary['asymptotic_roots_used'],bool), 'boundary kernel convergence summary conflict')
        require(summary['scope']=='requested_grid_and_all_production_boundary_callbacks', 'boundary evidence scope conflict')
        number(summary['min_time_s'],'min callback time',nonnegative=True)
        require(number(summary['max_time_s'],'max callback time',nonnegative=True)>=r.time_s[-1], 'boundary callback coverage conflict')
        terms=summary['max_terms_used_by_axis']
        require(isinstance(terms,tuple) and len(terms)==3,'boundary terms require xyz slots')
        for flag,used in zip(r.model.geometry.active_dimensions,terms):
            require(integer(used,'boundary terms')<=r.solver_config.max_terms and ((used>0)==flag),'boundary term budget/active face conflict')
        if unresolved:require('SERIES_TRUNCATION_UNRESOLVED' in codes,'internal boundary series caution hidden')
        if summary['asymptotic_roots_used']:require('LARGE_BI_ROOT_APPROXIMATION' in codes,'internal boundary root approximation hidden')
    parts = e['component_evidence']
    exact_fields(parts, result.contributions, 'component evidence')
    p = r.given_parameters
    evaluated=e['evaluated_parameters']
    mid=r.model.model_id
    dual=mid in {ModelID.M2_DUAL_HO_D,ModelID.M3_DUAL_HO_DK,ModelID.M8_DUAL_K}
    has_series=mid not in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}
    require(e['root_mode']==('legacy_full_roots' if has_series else 'not_applicable')
        and e['root_solver_policy']==(ROOT_SOLVER_POLICY if has_series else 'not_applicable'), 'root policy conflict')
    require(e['absolute_tolerance_usage']==('recorded_not_used_by_single_series_kernel' if has_series else 'not_used_by_analytic_surface_kernel'),
        'unused solver tolerance must not claim control of the forward calculation')
    background={k:v for k,v in p.items() if k.startswith('D_')} if mid in {ModelID.M1_SINGLE_DK,ModelID.M3_DUAL_HO_DK} else {}
    require(e['given_diffusion_background']==background, 'given D background conflict; no estimated D is implicit')
    for index,(key,c) in enumerate(parts.items()):
        exact_fields(c, SIMULATION_COMPONENT_FIELDS, 'component '+key)
        if mid in {ModelID.M2_DUAL_HO_D,ModelID.M3_DUAL_HO_DK}:
            slot=key[-1]; amplitude=p['A_'+slot]; role='parent_model_role_'+slot
            sign=(-1. if slot=='H' else 1.) if r.model.step_type=='hydration' else (1. if slot=='H' else -1.)
            if r.model.step_type=='custom':sign=getattr(r.model,'sign_'+slot)
            names=[f'D_{slot}_cm2_s']+([f'k_{slot}_cm_s'] if mid is ModelID.M3_DUAL_HO_DK else [])
        elif mid is ModelID.M8_DUAL_K:
            slot=int(key[-1]); amplitude=p[f'A_{slot}']; role=f'effective_input_mode_{slot}'
            sign=(-1. if slot==1 else 1.) if r.model.step_type=='hydration' else (1. if slot==1 else -1.)
            if r.model.step_type=='custom':sign=r.model.sign_H if slot==1 else r.model.sign_O
            names=[f'k_{slot}_cm_s']
        else:
            amplitude=p['amplitude'];sign=1.;role='single'
            names=[name for name in ('D_cm2_s','k_cm_s') if name in p]
        require(c['role']==role and c['sign']==sign and c['amplitude']==amplitude, 'component role/sign/amplitude conflict')
        require(c['kinetics']=={name:evaluated[name] for name in names}, 'component kinetic input conflict')
        F=vector(c['remaining_fraction'],'remaining fraction')
        progress=vector(c['completion_fraction'],'completion fraction')
        ideal=vector(c['ideal_completion_fraction'],'ideal completion fraction')
        require(len(F)==len(progress)==len(ideal)==len(r.time_s),'component omits requested times')
        # k-only uses -expm1 for small times; subtracting from 1 loses its
        # sub-epsilon completion. Never replace the accurate production array.
        close_vectors(F,tuple(1.-v for v in progress),'remaining/completion identity',atol=2e-16,rtol=0)
        close_vectors(result.contributions[key],tuple(sign*amplitude*v for v in progress),'signed component identity',atol=0,rtol=0)
        if has_series:
            require(c['surface_rates'] is None and c['speed_label'] is None,'series component cannot claim analytic surface rates')
            _validate_series(r,c, evaluated, names, codes)
        else:
            require(c['axis_results'] is None and c['kernel_diagnostics'] is None, 'k-only cannot claim diffusion/Bi diagnostics')
            rates=c['surface_rates']
            exact_fields(rates,('geometry_factor_cm_inv','rate_s_inv','tau_s','production_kernel_version','exposure_rule'),'surface rate evidence')
            g=fsum(2./d for d,a in zip(r.model.geometry.full_dims_cm,r.model.geometry.active_dimensions) if a)
            close_vectors((rates['geometry_factor_cm_inv'],),(g,),'two-face surface geometry',atol=0,rtol=1e-14)
            rate=number(rates['rate_s_inv'],'rate',positive=True)
            close_vectors((rate,),(g*evaluated[names[0]],),'given surface rate',atol=0,rtol=1e-14)
            close_vectors((number(rates['tau_s'],'tau',positive=True),),(1./rate,),'surface tau',atol=0,rtol=1e-14)
            require(rates['production_kernel_version']==MODEL_VERSION and rates['exposure_rule']==EXPOSURE_RULE,'surface kernel convention conflict')
            if mid is ModelID.M8_DUAL_K:
                other='component_2' if key=='component_1' else 'component_1'
                other_rate=parts[other]['surface_rates']['rate_s_inv']
                require(c['speed_label']==('equal_rate' if rate==other_rate else 'effective_fast' if rate>other_rate else 'effective_slow'), 'effective speed/input slot mismatch')
            else:require(c['speed_label'] is None,'single mode has no relative speed label')
        if r.boundary.mode=='ideal_step':
            close_vectors(progress,ideal,'ideal response identity',atol=0,rtol=0)
        if amplitude==0:require('ZERO_AMPLITUDE' in codes,'zero component caution hidden')
        for i,t in enumerate(r.time_s):
            if t==origin:require(progress[i]==0.,'production initial condition conflict')
    if not dual:
        c=parts['component_1']
        for name in ('remaining_fraction','completion_fraction','axis_results','kernel_diagnostics'):
            require(e[name]==c[name],'single/component evidence conflict')
    else:
        require(all(e[name] is None for name in ('remaining_fraction','completion_fraction','axis_results','kernel_diagnostics')),
            'dual total has no unique unit-amplitude remaining fraction')
    baseline=p['y0'] if dual else p['baseline']
    # Match production addition order, especially under opposing contributions.
    expected=tuple(baseline+result.contributions['component_H'][i]+result.contributions['component_O'][i]
        if mid in {ModelID.M2_DUAL_HO_D,ModelID.M3_DUAL_HO_DK} else
        baseline+result.contributions['component_1'][i]+result.contributions['component_2'][i]
        if mid is ModelID.M8_DUAL_K else baseline+result.contributions['component_1'][i] for i in range(len(r.time_s)))
    close_vectors(result.signal_clean,expected,'production common baseline identity',atol=0,rtol=0)
    if r.noise_sd == 0:
        require(result.signal_observed==result.signal_clean,'noise-free observed curve conflict')
    require(e['limit_signals']==dict(initial_time_s=origin,initial_signal=baseline,
        asymptotic_signal=baseline+sum(c['sign']*c['amplitude'] for c in parts.values()),
        asymptotic_basis='conditional_declared_final_gas' if r.boundary.mode=='measured_history' else 'given_parameters_ideal_or_flush_limit'),
        'initial/asymptotic given-parameter limit conflict')
    paths=e['nonfinite_diagnostic_paths']
    require(isinstance(paths,tuple) and len(set(paths))==len(paths),'nonfinite paths must be unique')
    for path in paths:text(path,'nonfinite diagnostic path')
    recorded=[]
    def tags(value,path):
        from collections.abc import Mapping
        if isinstance(value,Mapping):
            if set(value)=={'nonfinite'}:
                require(value['nonfinite'] in {'nan','positive_infinity','negative_infinity'},'unknown nonfinite tag')
                recorded.append(path)
            else:
                for key,v in value.items():tags(v,path+'.'+key)
        elif isinstance(value,tuple):
            for i,v in enumerate(value):tags(v,f'{path}[{i}]')
    for key,c in parts.items():
        prefix='' if len(parts)==1 else 'component_evidence.'+key+'.'
        tags(c['axis_results'],prefix+'axis_results')
        tags(c['kernel_diagnostics'],prefix+'kernel_diagnostics')
    tags(bd,'boundary_diagnostics')
    require(set(recorded)==set(paths),'nonfinite diagnostic path ledger conflict')
    unexpected=[p for p in paths if not (r.time_s[0]==0 and p.endswith('.diagnostics.truncation_error_by_time[0]'))]
    if unexpected:require('NONFINITE_PRODUCTION_DIAGNOSTIC' in codes,'unexpected nonfinite diagnostic caution hidden')


def _validate_series(r,c,evaluated,names,codes):
    axes=c['axis_results']; n=len(r.time_s)
    require(isinstance(axes,tuple) and len(axes)==3,'three ordered axis results required')
    active=[]
    for dim,flag,axis in zip(r.model.geometry.full_dims_cm,r.model.geometry.active_dimensions,axes):
        require((axis is not None)==flag,'active/inactive axis evidence conflict')
        if axis is None:continue
        exact_fields(axis,('remaining_fraction','n_terms_used','coefficient_sum','diagnostics'),'series axis')
        values=vector(axis['remaining_fraction'],'axis remaining fraction')
        require(len(values)==n,'axis omits requested times')
        require(integer(axis['n_terms_used'],'n_terms_used',minimum=1)<=r.solver_config.max_terms,'series exceeded budget')
        number(axis['coefficient_sum'],'coefficient_sum')
        d=axis['diagnostics']
        require(d['full_dim_cm']==dim and d['half_dim_cm']==dim/2 and d['term_tol']==r.solver_config.relative_tolerance
            and d['max_terms']==r.solver_config.max_terms and d['normalization_applied'] is False
            and isinstance(d['converged'],bool),'axis dimensions/solver convention conflict')
        knames=[name for name in names if name.startswith('k_')]
        if knames:
            D=evaluated[next(name for name in names if name.startswith('D_'))]
            close_vectors((d['bi'],),(evaluated[knames[0]]*(dim/2)/D,),'given-parameter Bi',atol=0,rtol=1e-14)
        else:require('bi' not in d,'D-only cannot contain an invented finite k/Bi')
        if not d['converged']:require('SERIES_TRUNCATION_UNRESOLVED' in codes,'unconverged series caution hidden')
        if d.get('asymptotic_roots_used'):require('LARGE_BI_ROOT_APPROXIMATION' in codes,'large-Bi approximation caution hidden')
        active.append(values)
    kd=c['kernel_diagnostics']
    require(tuple(kd['full_dims_cm'])==r.model.geometry.full_dims_cm
        and tuple(kd['active_dimensions'])==r.model.geometry.active_dimensions and kd['normalization_applied'] is False
        and kd['all_axes_converged']==all(a['diagnostics']['converged'] for a in axes if a is not None),'kernel geometry/convergence conflict')
    # Nonideal response includes boundary composition. Its material axis product
    # must never be passed off as the driven signal itself.
    if r.boundary.mode=='ideal_step':
        close_vectors(c['remaining_fraction'],tuple(prod(a[i] for a in active) for i in range(n)),
            'rectangular axis product',atol=2e-15,rtol=2e-15)
