"""Explicit frozen figure capture, published Batch provenance and worker events."""
from dataclasses import replace
from pyecr.gui.state import GuiPage
from pyecr.report.figure_catalog import capture_source,FigureSource,thaw,freeze

BATCH_CONTEXT_VERSION='batch-figure-sources/1'

class FigureExportControllerMixin:
    def _remember_figure_batch_source(self,result,entries,template,mode):
        source=capture_source(result,entries=entries,template=template,label='Batch' if mode=='independent' else 'Global')
        existing=self.analysis_session.project_metadata.get('figure_source_contexts',dict(version=BATCH_CONTEXT_VERSION,records={}))
        if existing.get('version')!=BATCH_CONTEXT_VERSION:raise ValueError('Unknown Batch figure source context version')
        records=dict(existing['records']);records[mode]=source.to_dict()
        self.analysis_session.project_metadata['figure_source_contexts']=dict(version=BATCH_CONTEXT_VERSION,records=records)

    def capture_batch_figure_source(self,result,*,scope='current'):
        context=self.batch_result_context(result)
        if context is not None:
            return capture_source(result,entries=context[1],template=context[2],scope=scope)
        from pyecr.services.batch_analysis import BatchAnalysisResult
        mode='independent' if isinstance(result,BatchAnalysisResult) else 'global'
        saved=self.analysis_session.project_metadata.get('figure_source_contexts')
        if saved is not None:
            if saved.get('version')!=BATCH_CONTEXT_VERSION:raise ValueError('Unknown Batch figure source context version')
            value=saved['records'].get(mode)
            if value is not None:
                source=FigureSource.from_dict(value)
                if freeze(thaw(source.payload)['result'])!=freeze(result.to_dict()):raise ValueError('Stored Batch figure observations/result identity conflict')
                from pyecr.plotting.view_state import freeze_views
                return replace(source,scope=scope,view_state=freeze_views(self.analysis_session.project_metadata.get('plot_views')))
        # Old projects lack these original inputs; summary figures remain valid.
        return capture_source(result,scope=scope)

    def figure_export_choices(self):
        """Labels/capture closures only; enumerating this menu never captures/runs."""
        choices=[];metadata=self.analysis_session.project_metadata
        def add(key,label,capture,guard=lambda:True):choices.append((key,label,capture,guard))
        def view(source):
            from pyecr.plotting.view_state import freeze_views
            return replace(source,view_state=freeze_views(metadata.get('plot_views')))
        session=self.analysis_session
        if self.can_capture_snapshot:
            fit=session.fit_result;revision=session.current_revision
            add('analysis','Analysis — current',lambda:view(capture_source(self.capture_snapshot(include_export_context=True))),
                lambda:self.analysis_session is session and session.fit_result is fit and not session.fit_stale and session.current_revision==revision)
        if self.session.report_snapshot is not None:
            saved=self.session.report_snapshot
            add('analysis_history','Analysis — explicitly captured history',lambda saved=saved:capture_source(saved,scope='historical',view_state=saved.export_context.get('project_metadata',{}).get('plot_views')))
        for mode,result in (('independent',self.session.batch_result),('global',self.session.global_result)):
            if result is not None:
                module='Batch' if mode=='independent' else 'Global'
                add(mode,module+' — stored result',lambda result=result:view(self.capture_batch_figure_source(result,scope='historical')))
        state=self.session.pair_design_state
        if not state.empty:add('pair','Pair — stored comparison and planning',lambda:view(capture_source(state,scope='historical')))
        if self.session.dct_result is not None:
            result=self.session.dct_result
            def legacy():
                from pyecr.project.interpretation_archive import capture_interpretations,legacy_targets
                from pyecr.services.dct_context_service import DctContextStore
                from pyecr.contracts import canonical_sha256,DctContextReview
                full=self.capture_dct_interpretations(self.capture_dct_archive())
                interpretations=None
                if full is not None:
                    key=canonical_sha256(result.to_dict());targets=legacy_targets(full)
                    if key in targets:
                        store=DctContextStore();store.history=[DctContextReview.from_dict(v) for v in full['history'] if v['target_sha256']==key]
                        selected=full['selected'].get(key)
                        store.selected={key:r for r in store.history if r.content_sha256==selected}
                        interpretations=capture_interpretations(store,None,legacy_targets={key:targets[key]})
                return view(capture_source(result,scope='historical',interpretations=interpretations))
            add('monotonic','DCT — stored Monotonic',legacy)
        from pyecr.project.dct_archive import DctArchive
        for component,service in ((False,self.signed_dct_service),(True,self.component_dct_service)):
            for job in service.history:
                payload=job.bundle if component else job.analysis
                if payload is None:continue
                def captured(job=job,component=component):
                    archive=DctArchive(component_jobs=(job,)) if component else DctArchive(whole_jobs=(job,))
                    return view(capture_source(archive,scope='historical',label=('Components ' if component else 'Signed ')+job.job_id,
                        interpretations=self.capture_dct_interpretations(archive,scoped=True)))
                add('dct_'+job.job_id,('DCT Components — ' if component else 'DCT Signed — ')+job.job_id+' ('+job.status+')',captured)
        for run in self.simulation.history:
            if run.result is not None:
                add('simulation_'+run.job_id,'Simulation — '+run.request.name+' ('+run.job_id+')',lambda run=run:view(capture_source(run.result,scope='historical',label=run.request.name)))
        return choices

    def start_figure_export(self,sources,destination,*,options=None,source_guards=()):
        if getattr(self,'_figure_export_task_id',None):raise RuntimeError('Figure export already running')
        if self.state.tasks[GuiPage.REPORT].status=='running':raise RuntimeError('Wait for the active report export')
        if self._dct_export_task_id is not None or self._simulation_export_task_id is not None:raise RuntimeError('Wait for the active module export')
        from pyecr.report.figure_bundle import export_figure_bundle,FigureExportOptions
        from pyecr.services.analysis import CancellationToken
        sources=tuple(sources);guards=tuple(source_guards);token=CancellationToken()
        def guard():
            for predicate in guards:
                if not predicate():raise RuntimeError('Figure source changed; output was not published')
            for source in sources:FigureSource.from_dict(source.to_dict())
        guard();settings=options or FigureExportOptions()
        task=self.runner.submit(GuiPage.REPORT,lambda progress:export_figure_bundle(sources,destination,options=settings,
            cancel_check=token.raise_if_cancelled,before_publish=guard,progress_callback=progress))
        self._figure_export_task_id=task;self._cancellation_tokens[task]=token
        self.figure_export_status='Queued frozen figures';self.figure_export_phase='queued';self.figure_export_progress=0.;return task

    def cancel_figure_export(self):
        task=getattr(self,'_figure_export_task_id',None)
        if task:
            self._cancellation_tokens[task].cancel();self.figure_export_status='Cancelling; previous complete output retained'

    def _apply_figure_export_event(self,event):
        self.figure_export_phase=event.kind
        if event.progress is not None:self.figure_export_progress=event.progress
        self.figure_export_status=event.kind+': '+event.message
        if event.kind=='error':self.figure_export_error_trace=event.traceback_text or ''
        if event.kind in ('completed','cancelled','error'):
            if event.kind=='completed':
                self.figure_export_result=event.result;self.figure_export_status='Verified figures: '+str(event.result.root)
            self._cancellation_tokens.pop(event.task_id,None);self._figure_export_task_id=None
