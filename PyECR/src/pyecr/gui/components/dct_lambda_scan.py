"""Stored regularization path viewer shared by monotonic and Signed DCT."""
from math import log10
from dataclasses import replace
from pyecr.gui.theme import TOKENS
from pyecr.gui.dct_scan import monotonic_scan, signed_scan, projected_reconstruction, displayed_candidates


class DctLambdaScanMixin:
    def _build_lambda_scan_window(self):
        d = self._require_dpg()
        with d.window(label='DCT regularization scan', tag='pyecr_dct_scan_window', show=False, width=1060, height=740,on_close=self._dock_lambda_scan):
            with d.group(tag='pyecr_dct_scan_body'):
                self._build_lambda_scan_content()
        self._register_owned_root('pyecr_dct_scan_window')
        self._dock_lambda_scan()

    def _build_lambda_scan_content(self):
        d=self._require_dpg()
        with d.group():
            d.add_text('Stored spectra: * selected; > viewed. Browsing does not run a solve.', tag='pyecr_dct_scan_intro',wrap=960)
            d.add_text('',tag='pyecr_dct_scan_scope',wrap=960)
            d.add_combo((),tag='pyecr_dct_scan_choice',width=-1,callback=self._change_lambda_scan)
            with d.group(horizontal=True):
                d.add_checkbox(label='Overlay all stored spectra',tag='pyecr_dct_scan_all',default_value=True,callback=self._change_lambda_scan)
                d.add_button(label='Use viewed lambda in Manual draft',tag='pyecr_dct_scan_manual',callback=self._scan_lambda_to_draft)
            for name,title,xlabel,ylabel in (
                    ('spectrum','Whole signal: spectra at every stored lambda','log10(tau / s)','chi / decade'),
                    ('score','GCV curve','log10(lambda)','GCV score'),
                    ('reconstruction','Reconstruction at the viewed lambda','Time (s)','Normalized signal')):
                with d.plot(label=title,tag='pyecr_dct_scan_'+name+'_plot',width=-1,height=410 if name=='spectrum' else 210):
                    d.add_plot_legend(location=d.mvPlot_Location_East)
                    d.add_plot_axis(d.mvXAxis,label=xlabel,tag='pyecr_dct_scan_'+name+'_x')
                    d.add_plot_axis(d.mvYAxis,label=ylabel,tag='pyecr_dct_scan_'+name+'_y')
                if name=='spectrum':d.add_text('',tag='pyecr_dct_scan_note',wrap=960,color=TOKENS.colors.warning)
            d.add_group(tag='pyecr_dct_scan_table')
    def _dock_lambda_scan(self,*args):
        d=self._require_dpg()
        if d.does_item_exist('pyecr_dct_scan_inline') and d.does_item_exist('pyecr_dct_scan_body'):
            d.move_item('pyecr_dct_scan_body',parent='pyecr_dct_scan_inline')
            d.configure_item('pyecr_dct_scan_window',show=False)
            from pyecr.gui.components.compact_panels import content_width
            for name in ('intro','scope','note'):
                tag='pyecr_dct_scan_'+name
                d.configure_item(tag,wrap=content_width(d,tag))

    def _lambda_scan_target(self):
        if self._dct_component_mode():
            job=self._selected_component_job()
            analysis=job.bundle.overall_analysis if job and job.bundle else None
            def factory():
                scan=signed_scan(analysis)
                return replace(scan,scope='WHOLE SIGNAL / GLOBAL lambda scan. Component shapes keep their separately declared Manual lambda. '+scan.scope)
            return ('component_whole',None if job is None else job.job_id,id(analysis)),factory
        if self._dct_signed_mode():
            if getattr(self,'_prior_evidence_active',lambda:False)():
                from pyecr.gui.prior_evidence import prior_scan
                archive=self._prior_archive
                return ('experimental_prior',id(archive)),lambda:prior_scan(archive)
            job = self._selected_signed_job()
            analysis = job.analysis if job else None
            if job and job.request.convex_options is not None:
                from pyecr.gui.convex_dct import convex_scan
                return ('convex',job.job_id,id(analysis)),lambda:convex_scan(analysis)
            return ('signed', False, None if job is None else job.job_id, id(analysis)), lambda: signed_scan(analysis)
        result = self.controller.session.dct_result
        return ('monotonic', id(result)), lambda: monotonic_scan(result)

    def _show_lambda_scan(self,*args):
        d = self._require_dpg()
        self._refresh_lambda_scan()
        d.move_item('pyecr_dct_scan_body',parent='pyecr_dct_scan_window')
        width=max(620,min(1100,d.get_viewport_client_width()-40))
        d.configure_item('pyecr_dct_scan_window',show=True,width=width,
                         height=max(480,min(820,d.get_viewport_client_height()-40)))
        for tag in ('intro','scope','note'):d.configure_item('pyecr_dct_scan_'+tag,wrap=width-36)
        d.set_y_scroll('pyecr_dct_scan_window',0)
        self._change_lambda_scan()

    def _refresh_lambda_scan(self):
        d = self._require_dpg()
        if not d.does_item_exist('pyecr_dct_scan_window'): return
        from pyecr.gui.components.compact_panels import content_width
        width=content_width(d,'pyecr_dct_scan_intro')
        if width!=getattr(self,'_dct_scan_wrap_width',None):
            self._dct_scan_wrap_width=width
            for name in ('intro','scope','note'):d.configure_item('pyecr_dct_scan_'+name,wrap=width)
        key, factory = self._lambda_scan_target()
        if key != getattr(self,'_dct_scan_key',None):
            self._dct_scan_key = key
            self._dct_scan = factory()
            scan = self._dct_scan
            labels = [f'{i+1:02d}  lambda={c.lam:.6g}'+(('  [GCV selected]' if scan.score_name=='GCV' else '  [stored selected]') if c.lam==scan.selected else '') for i,c in enumerate(scan.candidates)]
            d.configure_item('pyecr_dct_scan_choice',items=labels)
            choice = (min(range(len(labels)),key=lambda i:abs(scan.candidates[i].lam-scan.selected)) if scan.selected is not None else 0) if labels else None
            d.set_value('pyecr_dct_scan_choice',labels[choice] if choice is not None else '')
            self._dct_scan_labels = labels
            self._dct_scan_render_key = None
            self._change_lambda_scan()
        available = bool(self._dct_scan.candidates)
        d.configure_item('pyecr_dct_view_scan',enabled=available)
        d.configure_item('pyecr_dct_check_scope',show=not available)
        baseline = getattr(self,'_dct_scan_after_run',None)
        if baseline is not None and key != baseline and available:
            self._dct_scan_after_run = None
            self._show_lambda_scan()
        elif baseline is not None and not self.controller.dct_activity:
            self._dct_scan_after_run = None

    def _change_lambda_scan(self,*args):
        d = self._require_dpg()
        scan = getattr(self,'_dct_scan',None)
        if scan is None: return
        labels = self._dct_scan_labels
        choice = d.get_value('pyecr_dct_scan_choice')
        index = labels.index(choice) if choice in labels else 0
        overlay = bool(d.get_value('pyecr_dct_scan_all'))
        key = (self._dct_scan_key,index,overlay)
        if key == getattr(self,'_dct_scan_render_key',None): return
        self._dct_scan_render_key = key
        for tag in getattr(self,'_dct_scan_series',()):
            if d.does_item_exist(tag): d.delete_item(tag)
        tags = []
        def line(name,x,y,label,scatter=False):
            if not len(x): return
            tag = 'pyecr_dct_scan_series_'+str(len(tags))
            method = d.add_scatter_series if scatter else d.add_line_series
            method(list(x),list(y),label=label,parent='pyecr_dct_scan_'+name+'_y',tag=tag)
            tags.append(tag)
        valid = [c for c in scan.candidates if c.score is not None]
        line('score',[log10(c.lam) for c in valid],[c.score for c in valid],scan.score_name)
        d.configure_item('pyecr_dct_scan_score_y',label=scan.score_name)
        d.configure_item('pyecr_dct_scan_score_plot',label=scan.score_name+' curve')
        optimum = next((c for c in valid if c.lam==scan.selected),None)
        if optimum: line('score',[log10(optimum.lam)],[optimum.score],scan.score_name+' selected',True)
        candidate = scan.candidates[index] if scan.candidates else None
        if candidate:
            for c in displayed_candidates(scan,index,overlay):
                line('spectrum',[log10(t) for t in c.tau],c.chi,
                     f'lambda={c.lam:.3g}'+(' *' if c.lam==scan.selected else '')+(' >' if c is candidate else ''))
            reconstruction = projected_reconstruction(scan,candidate)
            line('reconstruction',scan.time,scan.observed,'Stored observations',True)
            if reconstruction: line('reconstruction',scan.time,reconstruction,f'Viewed lambda={candidate.lam:.5g}')
        self._dct_scan_series = tags
        d.set_value('pyecr_dct_scan_scope',scan.scope)
        note = '' if candidate is None else candidate.note
        if valid and optimum in (valid[0],valid[-1]):
            note += ('\nGCV minimum is at a tested boundary. Expand the candidate range before reading fine peak structure.' if scan.score_name=='GCV' and scan.manual_draft_allowed else '\n'+scan.score_name+' minimum is at a tested boundary; support beyond the tested range is unverified.')
        if candidate and not candidate.tau:
            note += '\nThe old archive contains scores only. Explicitly rerun GCV to retain candidate spectra; browsing never recomputes them.'
        d.set_value('pyecr_dct_scan_note',note)
        d.configure_item('pyecr_dct_scan_manual',enabled=scan.manual_draft_allowed and candidate is not None and bool(candidate.tau))
        self._dct_review_table('pyecr_dct_scan_table',('Lambda',scan.score_name,'RMSE','Selection / status'),
            [(f'{c.lam:.6g}','N/A' if c.score is None else f'{c.score:.6g}',
              'N/A' if c.rmse is None else f'{c.rmse:.6g}',('Selected; ' if c.lam==scan.selected else '')+c.note) for c in scan.candidates])
        for name in ('score','spectrum','reconstruction'):
            for axis in ('x','y'): d.fit_axis_data('pyecr_dct_scan_'+name+'_'+axis)

    def _scan_lambda_to_draft(self,*args):
        if not self._dct_scan.manual_draft_allowed:return
        d = self._require_dpg()
        index = self._dct_scan_labels.index(d.get_value('pyecr_dct_scan_choice'))
        value = self._dct_scan.candidates[index].lam
        d.set_value('pyecr_dct_lambda_mode','Manual')
        d.set_value('pyecr_dct_lambda',value)
        self._select_dct_lambda_mode(None,'Manual',None)
        d.set_value('pyecr_dct_scan_note',f'Manual draft lambda set to {value:.6g}. Run DCT explicitly to calculate a new result. Saved spectrum unchanged.')
