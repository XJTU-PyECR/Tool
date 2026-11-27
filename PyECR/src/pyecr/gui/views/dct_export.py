"""Explicit export of selected stored DCT evidence; drafts never execute."""
from pyecr.gui.theme import TOKENS

class DctExportViewMixin:
    def _build_dct_export_controls(self):
        dpg = self._require_dpg()
        with dpg.group(tag='pyecr_dct_export_controls', show=False):
            with dpg.collapsing_header(label='Export selected stored DCT run', default_open=False):
                dpg.add_text('CSV + JSON + report; all observations. Existing verified DCT bundles may be replaced.', wrap=TOKENS.dct.control_wrap)
                dpg.add_input_text(tag='pyecr_dct_export_destination', hint='Full destination directory', width=-1)
                dpg.add_checkbox(label='XLSX (15 digits; approximate)', tag='pyecr_dct_export_xlsx', default_value=False)
                dpg.add_checkbox(label='Auto-place legends (export only)',tag='pyecr_dct_export_legend_auto',default_value=False)
                with dpg.group(horizontal=True):
                    for fmt in ('png','svg','pdf'):
                        dpg.add_checkbox(label=fmt.upper(), tag='pyecr_dct_export_'+fmt, default_value=True)
                dpg.add_button(label='Export captured run', tag='pyecr_dct_export_run', callback=self._export_selected_dct)
                dpg.add_button(label='Cancel export', tag='pyecr_dct_export_cancel', callback=self._cancel_dct_export_widgets)
                dpg.add_text('', tag='pyecr_dct_export_status', wrap=TOKENS.dct.control_wrap)

    def _export_selected_dct(self, *args):
        dpg = self._require_dpg()
        try:
            component = self._dct_component_mode()
            job = self._selected_component_job() if component else self._selected_signed_job()
            if job is None:
                raise ValueError('Select a stored Signed or Components run in its history first')
            destination = str(dpg.get_value('pyecr_dct_export_destination')).strip()
            from pathlib import Path
            if not destination or not Path(destination).is_absolute():
                raise ValueError('Enter an absolute destination directory')
            formats=tuple(fmt for fmt in ('png','svg','pdf') if dpg.get_value('pyecr_dct_export_'+fmt))
            self.controller.start_dct_export(destination, job_id=job.job_id, component=component,
                include_xlsx=bool(dpg.get_value('pyecr_dct_export_xlsx')),
                figure_formats=formats,
                plot_views=self._plot_views.freeze() if formats else None,
                legend_policy='auto' if dpg.get_value('pyecr_dct_export_legend_auto') else 'follow')
        except Exception as exc:
            self.controller.dct_export_status = f'Export rejected: {type(exc).__name__}: {exc}'
        dpg.set_value('pyecr_dct_export_status', self.controller.dct_export_status)

    def _cancel_dct_export_widgets(self, *args):
        self.controller.cancel_dct_export()
        self._require_dpg().set_value('pyecr_dct_export_status', self.controller.dct_export_status)
