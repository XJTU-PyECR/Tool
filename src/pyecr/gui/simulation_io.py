"""Simulation export/load and reviewed Analysis transfer; no scientific algorithms."""
from pyecr.contracts.common import require
from pyecr.gui.state import GuiPage,TaskViewState
from pyecr.report.simulation_bundle import SimulationBundle,read_simulation_bundle,export_simulation_bundle
from pyecr.services.analysis import CancellationToken
from pyecr.services.workflow import AnalysisSession,WorkflowController
from pyecr.services.simulation_transfer import (SimulationSendPlan,prepare_simulation_send,target_identity,
    validate_simulation_origin,commit_simulation_send)


class SimulationIoControllerMixin:
    def _initialize_simulation_io(self):
        self._simulation_export_task_id=None;self._simulation_export_context=None
        self.simulation_export_status='No Simulation export requested'
        self.simulation_export_error_trace=''
        self.simulation_export_result=None;self.simulation_bundles={}

    def selected_simulation_bundle(self):
        run=self.simulation.selected
        if run is None or run.result is None:return None
        bundle=self.simulation_bundles.get(run.job_id)
        return bundle if bundle is not None and bundle.result.content_sha256==run.result.content_sha256 else None

    def start_simulation_export(self,destination,*,figure_formats=('png','svg','pdf'),include_xlsx=False,
                                legend_policy='follow',plot_views=None):
        require(not getattr(self,'_figure_export_task_id',None),'Wait for the active figure export')
        require(self._simulation_export_task_id is None,'Simulation export already running')
        run=self.simulation.selected
        require(run is not None and run.result is not None and run.result.state.ready,'select a complete Simulation result')
        token=CancellationToken();result=run.result
        from pyecr.report.export_options import ExportOptions
        from pyecr.plotting.view_state import freeze_views
        options=ExportOptions(plot_views=freeze_views(plot_views if plot_views is not None else self.analysis_session.project_metadata.get('plot_views')),
                              legend_policy=legend_policy)
        def execute(progress):
            return export_simulation_bundle(result,destination,figure_formats=figure_formats,include_xlsx=include_xlsx,
                options=options,
                cancel_check=token.raise_if_cancelled,progress_callback=progress)
        task_id=self.runner.submit(GuiPage.SIMULATION,execute)
        self._simulation_export_task_id=task_id;self._simulation_export_context=(run.job_id,result.content_sha256)
        self._cancellation_tokens[task_id]=token
        self.simulation_export_status='Queued captured Simulation export: '+run.job_id
        self.simulation_export_error_trace=''
        return task_id

    def cancel_simulation_export(self):
        token=self._cancellation_tokens.get(self._simulation_export_task_id)
        if token is None:return False
        self.simulation_export_status='Cancel export requested; an already committed new directory remains a completed export.'
        return token.cancel()

    def _apply_simulation_export_event(self,event):
        context=self._simulation_export_context
        if event.kind=='progress':
            self.simulation_export_status=f'{context[0]} · {event.progress:.0%}: '+event.message
            return
        if event.kind=='completed':
            bundle=event.result
            if not isinstance(bundle,SimulationBundle) or (bundle.result.request.job_id,bundle.result.content_sha256)!=context:
                self.simulation_export_status='Error: export identity conflict; no result attached to selected run'
            else:
                self.simulation_bundles[context[0]]=bundle;self.simulation_export_result=bundle
                self.simulation_export_status='Committed and readback verified: '+str(bundle.root)+' · '+context[0]
        else:
            self.simulation_export_status=event.kind+': '+event.message+'; previous complete exports retained'
            self.simulation_export_error_trace=event.traceback_text or ''
        self._cancellation_tokens.pop(event.task_id,None)
        self._simulation_export_task_id=None;self._simulation_export_context=None

    def open_simulation_bundle(self,root):
        bundle=read_simulation_bundle(root)
        self.simulation.restore_result(bundle.result)
        self.simulation_bundles[bundle.result.request.job_id]=bundle
        self.simulation_export_status='Explicitly loaded verified Simulation bundle: '+str(bundle.root)
        return bundle

    def prepare_simulation_send(self,channel,target_action):
        self._ensure_no_running_tasks()
        bundle=self.selected_simulation_bundle()
        require(bundle is not None,'Export or open this Simulation bundle first; Send uses its verified persistent CSV source')
        return prepare_simulation_send(bundle.root,channel,target_action,self.analysis_session)

    def send_simulated_data_to_analysis(self,plan):
        require(isinstance(plan,SimulationSendPlan),'reviewed SimulationSendPlan required')
        self._ensure_no_running_tasks()
        old=self.analysis_session
        require(plan.target_session is old and target_identity(old)==plan.target_sha256,
            'Analysis target changed after review; review the target impact again')
        d=plan.imported.dataset;y=validate_simulation_origin(d);origin=d.metadata['simulation_origin']
        selected=self.selected_simulation_bundle()
        require(selected is not None and selected.result.content_sha256==y.content_sha256,
            'Selected Simulation result changed after review; review Send again')
        bundle=read_simulation_bundle(origin['bundle_path_hint'])
        require(bundle.manifest['manifest_sha256']==origin['bundle_manifest_sha256'] and bundle.result==y,
            'Simulation source changed after review; prepare Send again')
        imported=commit_simulation_send(plan);d=imported.dataset
        # Construct the full replacement before changing any current state.
        fresh=AnalysisSession(data_revision=old.data_revision,import_revision=old.import_revision,
            config_revision=old.config_revision+1,physical_input_revision=old.physical_input_revision+1,
            fit_generation=old.fit_generation)
        workflow=WorkflowController(session=fresh,analysis_service=self.analysis_service)
        workflow.import_dataset(d)
        self.invalidate_report('Explicit Simulation data transfer; previous snapshot is historical')
        self.cancel_boundary_plot();self.cancel_explorer_preview()
        self.boundary_plot_result=None;self.explorer_plot_result=None
        self.session.analysis_session=fresh;self.workflow=workflow
        self.component_dct_service.bind_session(fresh);self._sync_signed_dct_inputs()
        self.session.project_snapshot=None;self.state.project_path=None
        self.state.latest_results.pop(GuiPage.PROJECT,None)
        self.state.tasks[GuiPage.PROJECT]=TaskViewState(message='Unsaved Simulation Analysis; use Save As')
        self.session.import_preview=imported.preview;self.session.import_result=imported
        self.session.dct_config=None;self.session.dct_result=None
        self.state.fit_presentation=None;self.state.diagnostics_presentation=None
        self.state.advanced_diagnostics.clear();self.state.error_map_result=None
        self.state.model_name=None;self.state.dataset_name=str(d.metadata['dataset_name'])
        self.state.time_unit='s';self.state.signal_unit=str(d.metadata['signal_unit'])
        for page in (GuiPage.MODEL,GuiPage.FIT,GuiPage.DIAGNOSTICS,GuiPage.DCT):
            self.state.latest_results.pop(page,None);self.state.tasks[page]=TaskViewState(message='Explicit Simulation import; configure/run separately')
        self.state.latest_results[GuiPage.DATA]=imported
        self.state.tasks[GuiPage.DATA]=TaskViewState(status='completed',progress=1.,
            message=f'Explicitly imported {len(d.raw_time)} {origin["source_kind"]} rows; no Apply/Fit/Diagnostics')
        self.guided_refit_last_message='No Guided Refit has been run.'
        self.robustness_assessment=None;self._robustness_request=None
        return imported
