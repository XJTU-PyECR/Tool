"""Explicit scientific DCT tables, made only from captured arrays (no display grid)."""
import numpy as np
from pyecr.report.export_tables import ExportTable, json_value


def _table(name, columns, types, units, rows):
    rows = tuple(tuple(r) for r in rows)
    return ExportTable(name, tuple(columns.split()), tuple(types.split()), units,
        len(rows), lambda: iter(rows))


def _spectrum(name, result):
    unit = result.physical_unit
    return _table(name, 'tau_s log10_tau q_decade chi mass physical_chi physical_mass',
        'float float float float float float? float?',
        dict(tau_s='s', log10_tau='log10(s)', q_decade='decade', chi='1/decade', mass='1',
            physical_chi='unavailable' if unit is None else unit+'/decade', physical_mass=unit or 'unavailable'),
        zip(result.tau_s, result.log10_tau, result.quadrature_weights, result.chi,
            result.quadrature_mass, result.physical_chi or (None,)*len(result.chi),
            result.physical_mass or (None,)*len(result.chi), strict=True))


def _observations(name, result):
    s = result.source; unit = result.physical_unit
    n = len(s.time_s)
    return _table(name, 'raw_index time_s dct_input dct_reconstruction dct_residual physical_dct_reconstruction physical_dct_residual',
        'integer float float float float float? float?',
        dict(time_s='s', dct_input=s.signal_unit, dct_reconstruction=s.signal_unit, dct_residual=s.signal_unit,
            physical_dct_reconstruction=unit or 'unavailable', physical_dct_residual=unit or 'unavailable'),
        zip(s.raw_indices, s.time_s, s.signal, result.reconstruction, result.residuals,
            result.physical_reconstruction or (None,)*n, result.physical_residuals or (None,)*n, strict=True))


def _analysis(prefix, analysis):
    tables = []
    if analysis is None:
        return tables
    from pyecr.contracts.convex_dct import ConvexDctAnalysis
    if isinstance(analysis,ConvexDctAnalysis):
        if analysis.selected_result is not None:
            tables += [_spectrum(prefix+'_spectrum',analysis.selected_result),_observations(prefix+'_observations',analysis.selected_result)]
        for i,c in enumerate(analysis.candidates):
            if c.result.state.has_payload:
                tables += [_spectrum(f'{prefix}_l{i:03d}_spectrum',c.result),_observations(f'{prefix}_l{i:03d}_obs',c.result)]
        tables.append(_table(prefix+'_lambda','lambda discrepancy_score status iterations weighted_sse curvature_penalty absolute_mass_penalty edge_penalty',
            'float float? text integer float float float float',{'lambda':'penalty_multiplier','discrepancy_score':'log_SSE_distance'},
            ((c.result.selected_lambda,c.score,c.result.payload['status'],c.result.payload['iterations'],c.result.payload['weighted_sse'],c.result.payload['curvature_penalty'],c.result.payload['absolute_mass_penalty'],c.result.payload['edge_penalty']) for c in analysis.candidates)))
        tables.append(_table(prefix+'_mode','mode selection options_json score_semantics','text text json text',{},
            [(analysis.options.mode,analysis.options.selection,json_value(analysis.options.to_dict()),analysis.score_name+'; no nonlinear GCV')]))
        tables.append(_table(prefix+'_peaks','tau_s sign signed_area absolute_area width_decades stability warnings_json','float text float float float text json',
            dict(tau_s='s',signed_area='1',absolute_area='1',width_decades='decade'),
            ((p.candidate.position_s,p.direction,p.signed_area,p.absolute_area,p.candidate.width_decades,p.stability_status,json_value(p.warnings)) for p in analysis.peak_diagnostics)))
        return tables
    if analysis.selected_result is not None:
        tables += [_spectrum(prefix+'_spectrum', analysis.selected_result),
                   _observations(prefix+'_observations', analysis.selected_result)]
    for i, c in enumerate(analysis.candidates):
        if c.result.state.has_payload:
            tables += [_spectrum(f'{prefix}_l{i:03d}_spectrum', c.result),
                       _observations(f'{prefix}_l{i:03d}_obs', c.result)]
    for i, c in enumerate(analysis.comparisons):
        if c.result is not None and c.result.state.has_payload:
            tables += [_spectrum(f'{prefix}_s{i:03d}_spectrum', c.result),
                       _observations(f'{prefix}_s{i:03d}_obs', c.result)]
    tables.append(_table(prefix+'_lambda', 'lambda effective_df raw_effective_df gcv_score failure',
        'float float? float? float? text?', {'lambda':'penalty_multiplier', 'effective_df':'1', 'raw_effective_df':'1', 'gcv_score':'weighted_signal_squared'},
        ((c.result.selected_lambda if c.result.selected_lambda is not None else c.result.config.grid.regularization_lambda,
          c.effective_df, c.raw_effective_df, c.gcv_score, c.failure) for c in analysis.candidates)))
    tables.append(_table(prefix+'_peaks',
        'tau_s sign signed_area absolute_area width_decades stability warnings_json',
        'float text float float float text json', dict(tau_s='s', signed_area='1', absolute_area='1', width_decades='decade'),
        ((p.candidate.position_s, p.direction, p.signed_area, p.absolute_area, p.candidate.width_decades,
            p.stability_status, json_value(p.warnings)) for p in analysis.peak_diagnostics)))
    return tables


def build_dct_tables(archive):
    tables = []
    for prefix, job in records(archive):
        ds = job.request.dataset
        tables.append(_table(prefix+'_source_raw', 'raw_index raw_time raw_signal included',
            'integer float float boolean', dict(raw_time=ds.metadata.get('raw_time_unit', ds.metadata.get('time_unit', 'unknown')),
            raw_signal=ds.metadata.get('raw_signal_unit', ds.metadata['signal_unit'])),
            ((i, t, y, m) for i, (t, y, m) in enumerate(zip(ds.raw_time, ds.raw_signal, ds.mask, strict=True)))))
        source=ds.metadata.get('source_kind','experiment')
        source=source if source in {'simulation_clean','simulation_observed'} and ds.metadata.get('simulation_send') is not None else 'experiment'
        source_column='processed_'+source
        tables.append(_table(prefix+'_'+source, 'raw_index processed_time '+source_column,
            'integer float float', dict(processed_time=ds.metadata['time_unit'], **{source_column:ds.metadata['signal_unit']}),
            zip(np.flatnonzero(ds.mask), ds.processed_time, ds.processed_signal, strict=True)))
        if prefix.startswith('w'):
            tables += _analysis(prefix, job.analysis)
            continue
        b = job.bundle
        if b is None:
            continue
        tables += _analysis(prefix+'_whole', b.overall_analysis)
        for i, r in enumerate(b.independent_results, 1):
            if r is None:
                continue
            stem = f'{prefix}_component{i}'
            u = r.physical_source.signal_unit
            tables.append(_table(stem+'_summary',
                'component_key lambda B S_total common_scale_factor shape_area common_signed_area common_absolute_area peak_status',
                'text float float float float float float float text',
                {'lambda':'penalty_multiplier', 'B':u, 'S_total':u, 'common_scale_factor':'1',
                    'shape_area':'1', 'common_signed_area':'1', 'common_absolute_area':'1'},
                ((r.source.component_key, r.config.grid.regularization_lambda, r.B, r.S_total, r.B/r.S_total,
                  sum(r.mass), sum(r.common_mass), sum(abs(v) for v in r.common_mass), 'not_extracted_in_P09'),)))
            tables.append(_table(stem+'_spectrum',
                'tau_s log10_tau q_decade shape_chi shape_mass common_signed_chi common_signed_mass physical_chi physical_mass',
                'float float float float float float float float float',
                dict(tau_s='s', log10_tau='log10(s)', q_decade='decade', shape_chi='1/decade', shape_mass='1',
                    common_signed_chi='1/decade', common_signed_mass='1', physical_chi=u+'/decade', physical_mass=u),
                zip(r.tau_s, r.log10_tau, r.quadrature_weights, r.chi, r.mass, r.common_chi, r.common_mass,
                    np.asarray(r.chi)*r.B, np.asarray(r.mass)*r.B, strict=True)))
            tables.append(_table(stem+'_obs',
                'raw_index time_s model_component remaining_shape shape_reconstruction shape_residual physical_component_dct physical_component_residual',
                'integer float float float float float float float',
                dict(time_s='s', model_component=u, remaining_shape='1', shape_reconstruction='1', shape_residual='1',
                    physical_component_dct=u, physical_component_residual=u),
                zip(r.source.raw_indices, r.source.time_s, r.physical_source.signal, r.source.signal,
                    r.reconstruction, r.residuals, r.physical_source.component_terminal_value+r.B*np.asarray(r.reconstruction),
                    r.B*np.asarray(r.residuals), strict=True)))
        complete = bool(b.component_time_reconstruction)
        n = len(b.parent_source.time_s); norm = b.parent_source.normalization
        combined = b.component_time_reconstruction if complete else (None,)*n
        residual = b.component_time_residuals if complete else (None,)*n
        observed=source+'_Y';difference=source+'_minus_component_dct'
        tables.append(_table(prefix+'_comparison',
            'raw_index time_s '+observed+' physical_fit_Y delta component_dct_Y component_approx_residual physical_fit physical_component_dct '+difference,
            'integer float float float float float? float? float float? float?',
            dict(time_s='s', **{observed:'1'}, physical_fit_Y='1', delta='1', component_dct_Y='1',
                component_approx_residual='1', physical_fit=norm.spec.signal_unit, physical_component_dct=norm.spec.signal_unit,
                **{difference:'1'}),
            ((idx, t, y, fit, b.delta, c, e, pf, None if c is None else norm.reference_value+norm.scale_value*c,
                None if c is None else y-c) for idx,t,y,fit,c,e,pf in zip(b.parent_source.raw_indices,
                b.parent_source.time_s, b.parent_source.signal, b.normalized_parent_fit, combined, residual,
                b.component_capture.fit_total, strict=True))))
    return tuple(tables)


def records(archive):
    return tuple((f'w{i:03d}', j) for i,j in enumerate(archive.whole_jobs)) + tuple(
        (f'c{i:03d}', j) for i,j in enumerate(archive.component_jobs))
