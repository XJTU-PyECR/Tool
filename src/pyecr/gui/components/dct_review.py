"""Read-only review of a stored DCT result. No inversion or model evaluation."""
from dataclasses import dataclass
from math import log10

from pyecr.gui.theme import TOKENS, SECTION_FONT_TAG


def number(value):
    return 'N/A' if value is None else f'{value:.5g}'


PEAK_STATES = {'not_checked': 'Not checked', 'incomplete': 'Incomplete',
               'unstable': 'Unstable', 'stable_conditional': 'Stable in listed checks'}
RESOLUTION_STATES = {
    'not_evaluated': 'Resolution not assessed',
    'no_resolvable_peak': 'No reportable peak',
    'single_candidate_not_process_count': 'One candidate; process count unproven',
    'unresolved_overlap': 'Overlapping candidates; separation unresolved',
    'separated_candidates_conditional': 'Separated candidates, conditional on this analysis',
    'oscillatory_candidates_not_resolved': 'Oscillatory spectrum; individual peaks unresolved',
    'boundary_limited_candidates': 'Boundary-limited candidates',
    'low_information_no_resolution_claim': 'Low information; resolution unproven',
}


def primary_caution(analysis):
    """Prioritize scientific limitations instead of alphabetically first codes."""
    priority = (
        ('OSCILLATORY_SPECTRUM', 'Spectrum oscillates; individual peaks are unresolved.'),
        ('LOW_SNR_HEURISTIC', 'Low information; peak resolution is unproven.'),
        ('NO_RESOLVABLE_PEAK', 'No peak meets the reporting thresholds.'),
        ('UNRESOLVED_NEIGHBOURING_CANDIDATES', 'Neighbouring peaks overlap; separation is unresolved.'),
        ('GCV_LAMBDA_AT_BOUNDARY', 'GCV selected a search-boundary lambda; review the lambda range.'),
        ('RESIDUAL_STRUCTURE', 'Residual structure needs review before interpreting peaks.'),
        ('BASIN_TRUNCATED', 'A peak tail reaches the tau boundary; its area may be incomplete.'),
        ('BOUNDARY_PEAK', 'A peak is near the tau boundary; review the tau range.'),
        ('FLAT_TOP_PEAK', 'A flat maximum has no unique peak position.'),
    )
    if analysis is None or not analysis.flags:return ''
    for flag,message in priority:
        if flag in analysis.flags:return message
    if 'EXPERIMENTAL_CONVEX_MODE' in analysis.flags:
        return 'Experimental mode; weak or slow negative contributions and stability need review.'
    from pyecr.dct.signed_analysis import FLAG_EXPLANATIONS
    return FLAG_EXPLANATIONS.get(analysis.flags[0],str(analysis.flags[0]))


def peak_limits(warnings):
    labels = {'BOUNDARY_PEAK':'Tau boundary','WIDTH_TRUNCATED':'Clipped width',
              'BASIN_TRUNCATED':'Clipped area','FLAT_TOP_PEAK':'Flat maximum'}
    return ', '.join(labels[w] for w in dict.fromkeys(warnings) if w in labels) or 'Candidate'


def matched_area_change(peak, comparisons, index):
    """Descriptive normalized-basin sensitivity, never an uncertainty bound."""
    shifts = [100*abs(c.peak_diagnostics[c.matches[index]].signed_area/peak.signed_area-1)
              for c in comparisons if c.failure is None and c.matches[index] is not None]
    return max(shifts) if shifts else None


@dataclass(frozen=True)
class DctReview:
    title: str = 'No stored result'
    severity: str = 'muted'
    scope: str = 'Run DCT, then review the saved result here.'
    settings: str = ''
    resolution: str = ''
    peaks: tuple = ()
    peak_empty: str = 'No stored peak candidates.'
    lambda_summary: str = 'No stored regularization result.'
    lambda_note: str = 'Run DCT to record the tested lambda values.'
    lambda_rows: tuple = ()
    selected_score: tuple = ()
    lambda_plot: bool = False
    stability_summary: str = 'No stored stability checks.'
    stability_rows: tuple = ()
    checks: tuple = ()
    cautions: tuple = ()


def review_signed(analysis, *, component=False, job=None):
    """Describe stored evidence only; missing/failed checks never become passes."""
    if job is not None and not component and job.request.convex_options is not None and analysis is not None:
        from pyecr.gui.convex_dct import review_convex
        return review_convex(analysis,job)
    scope = ('Whole-data analysis only. Component Manual lambda sensitivity: not run; no component CI.'
             if component else 'Whole-data signed spectrum. Checks use this saved run, independent of current drafts.')
    if analysis is None or analysis.selected_result is None:
        reason = getattr(job, 'failure', None) or 'No usable stored spectrum. Run DCT or select a completed run.'
        return DctReview(title='No usable stored spectrum', severity='warning', scope=scope,
                         peak_empty=reason, lambda_note=reason)
    r = analysis.selected_result; grid = r.config.grid
    peaks = tuple((str(i), number(p.candidate.position_s), '+' if p.direction == 'positive' else '−',
                   number(p.signed_area), number(p.candidate.width_decades),
                   peak_limits(p.warnings+(('BOUNDARY_PEAK',) if p.boundary else ())+
                               (('WIDTH_TRUNCATED',) if p.width_truncated else ())))
                  for i,p in enumerate(analysis.peak_diagnostics, 1))
    stability = tuple((str(i), number(p.candidate.position_s), PEAK_STATES[p.stability_status],
                       f'{p.matched_comparisons}/{p.required_comparisons}',
                       number(p.max_position_shift_decades),number(matched_area_change(p,analysis.comparisons,i-1)))
                      for i,p in enumerate(analysis.peak_diagnostics, 1))
    candidates = tuple((number(c.result.selected_lambda), number(c.gcv_score),
                         'Selected' if c.result.selected_lambda == r.selected_lambda else '',
                         c.failure or 'Scored') for c in analysis.candidates)
    selected = next((c for c in analysis.candidates if c.result.selected_lambda == r.selected_lambda
                     and c.gcv_score is not None), None)
    comparisons = analysis.comparisons
    checks = tuple((c.label, 'Not run' if c.failure and c.failure.startswith('NOT_RUN') else
                    'Failed' if c.failure else 'Computed', c.failure or
                    f'{sum(v is not None for v in c.matches)}/{len(c.matches)} peaks matched')
                   for c in comparisons)
    failed = sum(bool(c.failure) for c in comparisons)
    stable = sum(p.stability_status == 'stable_conditional' for p in analysis.peak_diagnostics)
    stability_summary = (f'{stable}/{len(peaks)} candidates position-stable in all listed checks · '
                         f'{len(comparisons)-failed}/{len(comparisons)} checks computed'
                         if comparisons else 'Not checked — no sensitivity comparisons were requested.')
    if comparisons and not peaks:
        stability_summary = f'{len(comparisons)-failed}/{len(comparisons)} checks computed; no peak candidates to match.'
    from pyecr.dct.signed_analysis import FLAG_EXPLANATIONS
    cautions = tuple(FLAG_EXPLANATIONS.get(f, str(f)) for f in analysis.flags)
    history = bool(job is not None and (job.stale_reasons or job.status == 'stale'))
    if history:
        cautions = tuple(job.stale_reasons) + cautions
    mode = grid.lambda_mode
    valid = sum(c.gcv_score is not None for c in analysis.candidates)
    return DctReview(title='Historical result · review source' if history else
                     'Saved result · review cautions' if cautions else 'Saved result available',
        severity='warning' if cautions or history else 'success', scope=scope,
        settings=f'{len(r.source.signal)} observations · tau {grid.tau_min_s:g}–{grid.tau_max_s:g} s · '
                 f'{grid.n_tau} nodes · {mode.upper()} lambda {r.selected_lambda:.5g}\n'
                 f'Tau readout: grid maxima, step {log10(grid.tau_max_s/grid.tau_min_s)/(grid.n_tau-1):.3g} decade; '
                 'not a confidence interval.',
        resolution=RESOLUTION_STATES[analysis.resolution_status],
        peaks=peaks, peak_empty='No candidate meets the stored peak-reporting thresholds.',
        lambda_summary=f'{"GCV selected" if mode == "gcv" else "Manual"} lambda = {r.selected_lambda:.5g}'+
                       (' · search boundary' if 'GCV_LAMBDA_AT_BOUNDARY' in analysis.flags else ''),
        lambda_note=(f'{valid}/{len(candidates)} candidates scored. Lower GCV is preferred within the tested set.'
                     if mode == 'gcv' else 'Manual value retained. No automatic lambda search was performed.') +
                    ' GCV scores are not confidence intervals; correlated noise can affect the choice.',
        lambda_rows=candidates,
        selected_score=() if selected is None else (log10(r.selected_lambda), selected.gcv_score),
        lambda_plot=mode == 'gcv' and valid > 0,
        stability_summary=stability_summary, stability_rows=stability, checks=checks, cautions=cautions)


def review_monotonic(presentation):
    if presentation is None:
        return DctReview()
    r = presentation.result
    basins = {p['grid_index']:p for p in r.diagnostics.get('peak_readout',{}).get('basins',())}
    lambdas=() if r.gcv_lambdas is None else tuple(zip(r.gcv_lambdas,r.gcv_scores))
    selected=next(((log10(lam),float(score)) for lam,score in lambdas if lam==r.regularization_lambda),())
    return DctReview(title='Saved monotonic result', severity='warning' if r.resolution_warnings else 'success',
        scope='Whole-data monotonic spectrum. Peak candidates are time scales, not species assignments.',
        settings=f'{len(presentation.time_s)} observations · lambda {r.regularization_lambda:.5g} ({r.lambda_source})\n'
                 f'Tau readout: grid maxima, step {r.diagnostics.get("log10_tau_step",0):.3g} decade; not a confidence interval.'+
                 (' Nonnegative GCV is a tuning heuristic.' if r.lambda_source=='gcv' else ''),
        resolution='Candidate time scales; interpret resolution cautions below.',
        peaks=tuple((str(i), number(p.position_s), '+', number(p.area), number(p.width_decades),
                     peak_limits(basins.get(p.grid_index,{}).get('warnings',())))
                    for i,p in enumerate(r.candidate_timescales,1)),
        peak_empty='No stored candidate time scales.', cautions=tuple(r.resolution_warnings),
        lambda_summary=f'{r.lambda_source.upper()} lambda = {r.regularization_lambda:.5g}',
        lambda_note='Stored candidate scores only; use View lambda scan to compare retained spectra. Nonnegative GCV is a tuning heuristic, not a confidence interval.',
        lambda_rows=tuple((number(lam),number(score),'Selected' if lam==r.regularization_lambda else '','Stored') for lam,score in lambdas),
        selected_score=selected,lambda_plot=bool(lambdas))


class DctReviewMixin:
    def _dct_review_heading(self, text, **kwargs):
        d = self._require_dpg(); item = d.add_text(text, **kwargs)
        if d.does_item_exist(SECTION_FONT_TAG): d.bind_item_font(item, SECTION_FONT_TAG)
        return item

    def _build_dct_review_tabs(self):
        d = self._require_dpg()
        with d.tab(label='Summary & peaks', tag='pyecr_dct_evidence_tab', show=False):
            self._dct_review_heading('', tag='pyecr_dct_review_title')
            d.add_text('', tag='pyecr_dct_review_scope', wrap=TOKENS.wraps.page)
            d.add_text('', tag='pyecr_dct_review_settings', wrap=TOKENS.wraps.page, color=TOKENS.steps.secondary)
            self._dct_review_heading('', tag='pyecr_dct_review_resolution', wrap=TOKENS.wraps.page, color=TOKENS.colors.warning)
            d.add_separator()
            self._dct_review_heading('Peak candidates', tag='pyecr_dct_review_peak_heading')
            d.add_text('Area integrates each sign/valley basin; rejected small basins are not reassigned. Width is zero-baseline FWHM in log10 decades. Position stability does not establish area accuracy or a chemical process.',
                       wrap=TOKENS.wraps.page, tag='pyecr_dct_review_peak_help')
            d.add_text('', tag='pyecr_dct_review_peak_empty', wrap=TOKENS.wraps.page)
            d.add_group(tag='pyecr_dct_review_peaks')
            d.add_separator()
            self._dct_review_heading('Interpretation cautions')
            d.add_text('', tag='pyecr_dct_review_cautions', wrap=TOKENS.wraps.page)
            with d.collapsing_header(label='How to read the spectrum', default_open=False):
                from pyecr.gui.signed_dct import SIGN_LEGEND, DISPLAY_NOTICE
                d.add_checkbox(label='View restored input units',tag='pyecr_signed_dct_physical',default_value=False,show=False)
                d.add_text(SIGN_LEGEND,tag='pyecr_signed_dct_legend',show=False,wrap=TOKENS.wraps.page)
                d.add_text(DISPLAY_NOTICE,tag='pyecr_signed_dct_display_notice',show=False,wrap=TOKENS.wraps.page)
            with d.collapsing_header(label='Source, normalization & full run details', default_open=False,
                                     tag='pyecr_dct_run_details_header'):
                d.add_text('',tag='pyecr_dct_mode_notice',wrap=TOKENS.wraps.page)
                with d.group(horizontal=True):
                    d.add_button(label='Open full evidence',callback=self._show_dct_evidence)
                    d.add_button(label='Copy full evidence',callback=self._copy_dct_evidence)
                d.add_input_text(tag='pyecr_dct_details',default_value='',readonly=True,multiline=True,
                                 width=-1,height=230)
        with d.tab(label='Lambda',tag='pyecr_signed_dct_diagnostics_tab',show=False):
            d.add_text('',tag='pyecr_dct_check_scope',wrap=TOKENS.wraps.page)
            d.add_button(label='Expand whole lambda spectra',tag='pyecr_dct_expand_lambda',callback=self._show_lambda_scan)
            d.add_group(tag='pyecr_dct_scan_inline')
            self._dct_review_heading('1  Regularization choice')
            self._dct_review_heading('',tag='pyecr_dct_check_lambda',color=TOKENS.colors.csv)
            d.add_text('',tag='pyecr_dct_check_lambda_note',wrap=TOKENS.wraps.page)
            with d.collapsing_header(label='GCV curve / selected lambda',default_open=False,tag='pyecr_dct_gcv_curve_header'):
                with d.plot(label='GCV search · selected value marked',tag='pyecr_dct_lambda_plot',width=-1,height=230):
                    d.add_plot_axis(d.mvXAxis,label='log10(lambda)',tag='pyecr_dct_lambda_x_axis')
                    with d.plot_axis(d.mvYAxis,label='GCV score',tag='pyecr_dct_lambda_y_axis'):
                        d.add_line_series([],[],tag='pyecr_signed_dct_lambda_series')
                        d.add_scatter_series([],[],tag='pyecr_signed_dct_lambda_points')
                        d.add_scatter_series([],[],tag='pyecr_dct_selected_lambda',label='Selected lambda')
            with d.theme() as selected_theme:
                with d.theme_component(d.mvScatterSeries):
                    d.add_theme_color(d.mvPlotCol_MarkerFill,TOKENS.colors.warning,category=d.mvThemeCat_Plots)
                    d.add_theme_color(d.mvPlotCol_MarkerOutline,TOKENS.colors.warning,category=d.mvThemeCat_Plots)
                    d.add_theme_style(d.mvPlotStyleVar_MarkerSize,6,category=d.mvThemeCat_Plots)
            d.bind_item_theme('pyecr_dct_selected_lambda',selected_theme)
            with d.collapsing_header(label='All lambda candidates / failed scores',default_open=False):
                d.add_group(tag='pyecr_dct_review_lambdas')
            d.add_separator()
            self._dct_review_heading('2  Peak position stability')
            d.add_text('',tag='pyecr_dct_check_stability',wrap=TOKENS.wraps.page,color=TOKENS.colors.warning)
            d.add_text('Position and area changes cover matched peaks only; missing matches still limit stability. Area uses the reported normalized scale, including S changes in terminal checks. These checks are not a confidence interval.',
                       tag='pyecr_dct_check_limits',wrap=TOKENS.wraps.page)
            d.add_group(tag='pyecr_dct_review_stability')
            with d.collapsing_header(label='Peak shift plot',default_open=False,tag='pyecr_dct_peak_shift_header'):
                with d.plot(label='Matched peak shifts',tag='pyecr_dct_stability_plot',width=-1,height=220):
                    d.add_plot_axis(d.mvXAxis,label='log10(tau / s)',tag='pyecr_dct_stability_x_axis')
                    with d.plot_axis(d.mvYAxis,label='Max matched shift (decade)',tag='pyecr_dct_stability_y_axis'):
                        d.add_scatter_series([],[],tag='pyecr_signed_dct_stability_series')
            with d.collapsing_header(label='Checks performed / missing checks',default_open=False):
                d.add_group(tag='pyecr_dct_review_checks')

    def _dct_review_table(self, tag, columns, rows):
        d=self._require_dpg(); d.delete_item(tag,children_only=True)
        if not rows: return
        widths=[max(140,min(360,30+11*max(len(str(title)),*(min(len(str(row[i])),30) for row in rows))))
                for i,title in enumerate(columns)]
        with d.table(parent=tag,header_row=True,resizable=True,policy=d.mvTable_SizingStretchProp,width=-1,
                     borders_innerH=True,pad_outerX=False,scrollX=True,scrollY=True,freeze_rows=1,
                     height=min(360,max(82,34*(len(rows)+1)+8)),inner_width=sum(widths)) as table:
            for label,width in zip(columns,widths):d.add_table_column(label=label,init_width_or_weight=width,width_fixed=True)
            for row in rows:
                with d.table_row():
                    for value,width in zip(row,widths):d.add_text(str(value),wrap=width-18)
        layouts=getattr(self,'_dct_table_widths',{})
        layouts[tag]=(table,tuple(widths));self._dct_table_widths=layouts

    def _refresh_dct_review(self):
        d=self._require_dpg()
        if not d.does_item_exist('pyecr_dct_review_title'): return
        component=self._dct_component_mode(); signed=self._dct_signed_mode()
        job=(self._selected_component_job() if component else self._selected_signed_job()) if signed else None
        result=self.controller.session.dct_result if not signed else None
        key=(id(job),id(result),signed,component)
        if key==getattr(self,'_dct_review_key',None):
            d.configure_item('pyecr_dct_lambda_plot',show=self._dct_review_model.lambda_plot)
            d.configure_item('pyecr_dct_gcv_curve_header',show=self._dct_review_model.lambda_plot)
            return
        if signed:
            analysis=(job.bundle.overall_analysis if job and job.bundle else None) if component else (job.analysis if job else None)
            view=review_signed(analysis,component=component,job=job)
        else: view=review_monotonic(result)
        if not signed:
            import math
            pairs=[] if result is None or result.result.gcv_lambdas is None else [(math.log10(lam),float(score)) for lam,score in zip(result.result.gcv_lambdas,result.result.gcv_scores) if math.isfinite(score)]
            values=[[p[0] for p in pairs],[p[1] for p in pairs]]
            d.set_value('pyecr_signed_dct_lambda_series',values)
            d.set_value('pyecr_signed_dct_lambda_points',values)
        self._dct_review_key=key
        self._dct_review_model=view; self._dct_review_job=job
        color=TOKENS.steps.secondary if view.severity=='muted' else getattr(TOKENS.colors,view.severity)
        for tag,value in (('review_title',view.title),('review_scope',view.scope),('review_settings',view.settings),
                          ('review_resolution',view.resolution),
                          ('review_peak_empty',view.peak_empty),('check_scope',view.scope),
                          ('check_lambda',view.lambda_summary),('check_lambda_note',view.lambda_note),
                          ('check_stability',view.stability_summary)):
            d.set_value('pyecr_dct_'+tag,value)
        d.configure_item('pyecr_dct_review_title',color=color)
        d.configure_item('pyecr_dct_check_scope',color=TOKENS.colors.warning if component else TOKENS.steps.secondary)
        d.set_value('pyecr_dct_review_peak_heading','Whole-data peak candidates' if component else 'Peak candidates')
        d.configure_item('pyecr_dct_review_peak_empty',show=not view.peaks)
        d.set_value('pyecr_dct_review_cautions','\n'.join('• '+v for v in view.cautions) or
                    'No stored caution flags. Peak count and sign do not identify species.')
        d.configure_item('pyecr_dct_review_cautions',color=TOKENS.colors.warning if view.cautions else TOKENS.steps.secondary)
        self._dct_review_table('pyecr_dct_review_peaks',('#','Tau (s)','Sign','Signed area','Width (decade)','Limits'),view.peaks)
        score_name=getattr(analysis,'score_name','GCV score') if signed else 'GCV score'
        convex=signed and getattr(analysis,'options',None) is not None
        d.configure_item('pyecr_dct_gcv_curve_header',label='Selection score / selected lambda' if convex else 'GCV curve / selected lambda')
        d.configure_item('pyecr_dct_lambda_plot',label=(analysis.options.mode+' · '+analysis.score_name) if convex else 'GCV search · selected value marked')
        self._dct_review_table('pyecr_dct_review_lambdas',('Lambda',score_name,'Selection','Status / reason'),view.lambda_rows)
        self._dct_review_table('pyecr_dct_review_stability',('#','Tau (s)','Position check','Matched checks','Max shift (decade)','Max area change (%)'),view.stability_rows)
        self._dct_review_table('pyecr_dct_review_checks',('Perturbation','Status','Result / reason'),view.checks)
        if not view.checks:
            d.add_text('No comparisons stored. Stability has not been assessed.',parent='pyecr_dct_review_checks',wrap=500)
        d.configure_item('pyecr_dct_lambda_plot',show=view.lambda_plot)
        d.configure_item('pyecr_dct_gcv_curve_header',show=view.lambda_plot)
        d.configure_item('pyecr_dct_peak_shift_header',show=any(row[4]!='N/A' for row in view.stability_rows))
        x,y=view.selected_score if view.selected_score else (None,None)
        d.set_value('pyecr_dct_selected_lambda',[[] if x is None else [x],[] if y is None else [y]])

    def _size_dct_review_tables(self, width):
        """Fill the available rail, with horizontal scrolling only when narrow."""
        d=self._require_dpg()
        key=(width,getattr(self,'_dct_review_key',None),getattr(self,'_dct_context_table_key',None),id(getattr(self,'_dct_peak_guide',None)))
        if key==getattr(self,'_dct_review_table_size',None):return
        self._dct_review_table_size=key
        inner=max(620,int(width)-2*TOKENS.dct.plot_gutter-12)
        for name in ('review_peaks','review_lambdas','review_stability','review_checks',
                     'context_process_table','context_axis_table','peak_guide_processes','peak_guide_support','peak_guide_matches'):
            for table in d.get_item_children('pyecr_dct_'+name,1) or ():
                if not d.get_item_info(table)['type'].endswith('mvTable'):continue
                layout=getattr(self,'_dct_table_widths',{}).get('pyecr_dct_'+name)
                if layout and layout[0]==table:
                    d.configure_item(table,inner_width=max(inner,sum(layout[1])))
                    continue
                d.configure_item(table,inner_width=inner)
                columns=d.get_item_children(table,0) or ()
                weights=[.35 if d.get_item_configuration(c)['label']=='#' else 1. for c in columns]
                total=sum(weights) or 1.
                for row in d.get_item_children(table,1) or ():
                    for index,item in enumerate(d.get_item_children(row,1) or ()):
                        if index<len(weights) and d.get_item_info(item)['type'].endswith('mvText'):
                            d.configure_item(item,wrap=max(40,min(240,int(inner*weights[index]/total)-16)))
