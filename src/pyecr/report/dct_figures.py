"""Scientific figures from stored full observation arrays; no recalculation."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pyecr.report.dct_tables import records


def write_dct_figures(archive, root, formats, options, check, *, emit=None):
    if root is not None:root.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        if not prefix.startswith('w') and job.request.capture.eligibility is not None:
            from .capability_evidence import caption
            fig.supxlabel(caption(job.request.capture.eligibility.to_dict()),fontsize=7)
        if emit is not None:
            try:emit(name,fig)
            finally:fig.clear()
            return
        FigureCanvasAgg(fig)
        from matplotlib.text import Text
        for item in fig.findobj(Text):
            item.set_fontfamily(options.font_family)
        fig.tight_layout()
        from pyecr.plotting.view_state import apply_axis_view
        if name.endswith('_whole_signed'):
            for axis,plot_id,ids in zip(fig.axes,('pyecr_dct_spectrum_plot','pyecr_dct_reconstruction_plot','pyecr_dct_residual_plot'),
                (('pyecr_signed_dct_positive','pyecr_signed_dct_negative','pyecr_signed_dct_zero'),('pyecr_dct_data_series','pyecr_dct_reconstruction_series'),('pyecr_dct_residual_series',))):
                apply_axis_view(axis,plot_id,options.plot_views,ids,legend_policy=options.legend_policy)
        elif name.endswith('_component_comparison'):
            component_ids=tuple('pyecr_component_dct_component_'+str(i) for i,c in enumerate(job.bundle.independent_results,1) if c is not None)
            for n,(axis,plot_id,ids) in enumerate(zip(fig.axes,('pyecr_dct_spectrum_plot','pyecr_dct_spectrum_plot','pyecr_dct_reconstruction_plot','pyecr_dct_residual_plot'),
                (component_ids,component_ids+('__whole_reference',),
                 ('pyecr_dct_data_series','pyecr_component_dct_parent_fit','pyecr_component_dct_delta','pyecr_dct_reconstruction_series'),
                 ('pyecr_component_dct_model_error','pyecr_dct_residual_series')))):
                # Independent shapes have a different y scale from the common
                # signed GUI comparison. Reuse its legend, never its axis range.
                apply_axis_view(axis,plot_id,options.plot_views,ids,ranges=n!=0,legend_policy=options.legend_policy)
        elif name.endswith('_lambda'):
            apply_axis_view(fig.axes[0],'pyecr_dct_lambda_plot',options.plot_views,log_coordinates=(True,False),legend_policy=options.legend_policy)
        elif name.endswith('_peak_stability'):
            apply_axis_view(fig.axes[0],'pyecr_dct_stability_plot',options.plot_views,legend_policy=options.legend_policy)
        try:
            for fmt in formats:
                check()
                fig.savefig(root/(name+'.'+fmt), format=fmt, dpi=options.image_dpi)
        finally:
            fig.clear()

    for prefix, job in records(archive):
        analysis = job.analysis if prefix.startswith('w') else (job.bundle.overall_analysis if job.bundle else None)
        if analysis is None or analysis.selected_result is None:
            continue
        r = analysis.selected_result
        fig = Figure(figsize=(options.width_in, options.height_in*1.8))
        ax, time, residual = fig.subplots(3, 1)
        ax.plot(r.log10_tau, np.maximum(r.chi,0), color='#e15a37', label='Positive signed density (1/decade)')
        ax.plot(r.log10_tau, np.minimum(r.chi,0), color='#3c91eb', label='Negative signed density (1/decade)')
        ax.axhline(0., color='gray', linewidth=.7, label='Zero')
        ax.set(xlabel='log10(tau / s)', ylabel='chi (1/decade)',
            title='Positive: above terminal; negative: below terminal')
        ax.legend(fontsize=8)
        source_label = {'simulation_clean':'Simulated clean','simulation_observed':'Simulated observations'}.get(r.source.source_kind.value,'Experiment')
        time.plot(r.source.time_s, r.source.signal, '.', label=source_label+' (all valid rows)')
        time.plot(r.source.time_s, r.reconstruction, label='Whole DCT reconstruction')
        time.set(xlabel='Time (s)', ylabel='Terminal deviation (1)'); time.legend(fontsize=8)
        residual.plot(r.source.time_s, r.residuals, '.', label=source_label+' - whole DCT')
        residual.axhline(0., color='gray', linewidth=.7)
        residual.set(xlabel='Time (s)', ylabel='Residual (1)'); residual.legend(fontsize=8)
        save(fig, prefix+'_whole_signed')
        fig = Figure(figsize=(options.width_in, options.height_in))
        ax = fig.subplots()
        convex=getattr(analysis,'options',None) is not None
        score=lambda c:c.score if convex else c.gcv_score
        candidates = [c for c in analysis.candidates if score(c) is not None]
        if convex:candidates.sort(key=lambda c:c.result.selected_lambda)
        ax.plot([c.result.config.grid.regularization_lambda for c in candidates],
            [score(c) for c in candidates], 'o-')
        if candidates and all(c.result.config.grid.regularization_lambda > 0 for c in candidates):
            ax.set_xscale('log')
        ax.set(xlabel='lambda (penalty multiplier)', ylabel=analysis.score_name if convex else 'GCV score', title=('Stored '+analysis.options.mode+' candidates; conditional evidence' if convex else 'Stored lambda candidates; GCV is not a CI'))
        save(fig, prefix+'_lambda')
        fig = Figure(figsize=(options.width_in, options.height_in))
        ax = fig.subplots()
        for p in analysis.peak_diagnostics:
            if p.max_position_shift_decades is not None:
                ax.plot(np.log10(p.candidate.position_s), p.max_position_shift_decades, 'o',
                    label=f'{p.direction}: {p.stability_status}')
        ax.set(xlabel='log10(tau / s)', ylabel='Max matched shift (decade)',
            title='Position sensitivity; missing controls are not zero uncertainty')
        if ax.lines:
            ax.legend(fontsize=8)
        save(fig, prefix+'_peak_stability')
        fig = Figure(figsize=(options.width_in, options.height_in))
        ax = fig.subplots()
        if r.physical_reconstruction is not None:
            norm = r.source.normalization
            observed = norm.reference_value + norm.scale_value*np.asarray(r.source.signal)
            ax.plot(r.source.time_s, observed, '.', label=source_label+' in restored input units')
            ax.plot(r.source.time_s, r.physical_reconstruction, label='DCT reconstruction in input units')
            ax.legend(fontsize=8)
        ax.set(xlabel='Time (s)', ylabel=r.physical_unit or 'Physical restoration unavailable')
        save(fig, prefix+'_physical_dct')
        if prefix.startswith('w'):
            continue
        b = job.bundle
        fig = Figure(figsize=(options.width_in, options.height_in*2))
        shape, common, time, residual = fig.subplots(4, 1)
        for i, c in enumerate(b.independent_results, 1):
            if c is None:
                continue
            shape.plot(c.log10_tau, c.chi, label=f'component_{i} independent shape')
            common.plot(c.log10_tau, c.common_chi, label=f'component_{i}: (B/S)*chi')
        simulated=b.parent_source.source_kind.value.startswith('simulation_')
        whole_label='Whole signed '+b.parent_source.source_kind.value if simulated else 'Whole signed experiment'
        observation_label={'simulation_clean':'Simulated clean Y','simulation_observed':'Simulated observations Y'}.get(b.parent_source.source_kind.value,'Experiment Y')
        common.plot(r.log10_tau, r.chi, label=whole_label)
        for a, title in ((shape, 'Independent model-derived shape; no independent noise samples'),
                (common, 'Common signed scale; different grids are never interpolated')):
            a.axhline(0., color='gray', linewidth=.7)
            a.set(xlabel='log10(tau / s)', ylabel='chi (1/decade)', title=title)
            a.legend(fontsize=7)
        t = b.parent_source.time_s
        time.plot(t, b.parent_source.signal, '.', label=observation_label)
        time.plot(t, b.normalized_parent_fit, label='Physical parent Fit Y')
        time.plot(t, [b.delta]*len(t), label='Explicit delta')
        if b.component_time_reconstruction:
            time.plot(t, b.component_time_reconstruction, label='delta + sum((B/S)*shape DCT)')
            residual.plot(t, b.component_time_residuals, label='Parent Fit - component DCT')
            residual.plot(t, np.asarray(b.parent_source.signal)-b.component_time_reconstruction,
                label=(observation_label+' - component DCT') if simulated else 'Experiment - component DCT')
        time.set(xlabel='Time (s)', ylabel='Terminal deviation (1)'); time.legend(fontsize=7)
        residual.set(xlabel='Time (s)', ylabel='Residual (1)')
        if residual.lines:
            residual.legend(fontsize=7)
        save(fig, prefix+'_component_comparison')
