"""User-controlled optional preprocessing preview, review, accept and unmask."""
from __future__ import annotations

import json
import traceback

import numpy as np

from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.preprocessing import ProcessingStep
from pyecr.preprocessing.optional import (
    preprocessing_comparison, preview_downsampling,
    preview_drift, preview_outliers,
)
from pyecr.gui.preprocessing_workflow import (
    FIELD_DOMAINS, optional_parameters, parse_indices, preview_evidence,
    reviewable_mask_indices,
)
from pyecr.preprocessing import PreprocessConfig

_EXAMPLES = {
    "drift_correction": {"method": "robust_linear", "reference_window": [90.0, 100.0], "scope": [0.0, 100.0], "anchor_time": 0.0},
    "outlier_mask": {"threshold": 6.0, "half_window": 3, "noise_floor": 0.0},
    "adaptive_downsample": {"target_points": 100, "early_until": 10.0, "late_power": 2.0},
}


class OptionalPreprocessingViewMixin:
    def _build_optional_preprocessing(self) -> None:
        dpg = self._require_dpg()
        with dpg.group(tag="pyecr_optional_controls"):
            with dpg.value_registry():
                dpg.add_string_value(tag="pyecr_optional_kind", default_value="drift_correction")
            with dpg.group(tag="pyecr_optional_typed_fields"):
                for kind, fields in (
                    ("drift_correction", ("drift_method", "reference_start", "reference_end", "scope_start", "scope_end", "anchor_time")),
                    ("outlier_mask", ("threshold", "half_window", "noise_floor")),
                    ("adaptive_downsample", ("target_points", "early_until", "late_power")),
                ):
                    with dpg.group(tag=f"pyecr_optional_form_{kind}"):
                        for field in fields:
                            if field == "drift_method":
                                control=dpg.add_combo(("Least squares", "Robust (Theil-Sen)"), label="Trend estimator", tag=f"pyecr_optional_field_{field}", default_value="Robust (Theil-Sen)", width=TOKENS.widths.field_wide)
                            else:
                                defaults = {"reference_start": "90", "reference_end": "100", "scope_start": "0", "scope_end": "100", "anchor_time": "0", "threshold": "6", "half_window": "3", "noise_floor": "0", "target_points": "100", "early_until": "10", "late_power": "2"}
                                labels={'reference_start':'Reference start','reference_end':'Reference end',
                                    'scope_start':'Correction start','scope_end':'Correction end','anchor_time':'Anchor time (optional)',
                                    'threshold':'Score threshold','half_window':'Half-window (points)',
                                    'noise_floor':'Noise floor','target_points':'Target points',
                                    'early_until':'Keep all points until','late_power':'Late-time spacing power'}
                                control=dpg.add_input_text(label=labels[field], tag=f"pyecr_optional_field_{field}", default_value=defaults[field], width=TOKENS.widths.field_wide)
                            with dpg.tooltip(parent=control):dpg.add_text(FIELD_DOMAINS[field],wrap=320)
                            dpg.add_text("", tag=f"pyecr_optional_error_{field}", wrap=TOKENS.wraps.rail,show=False,color=TOKENS.colors.error)
            dpg.add_input_text(label="Advanced recipe (copyable JSON)", tag="pyecr_optional_options", default_value=json.dumps(_EXAMPLES["drift_correction"]), readonly=True, multiline=True, height=TOKENS.heights.history, width=-1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Preview", tag="pyecr_optional_preview", callback=self._preview_optional)
                dpg.add_button(label="Accept", tag="pyecr_optional_accept", enabled=False, callback=self._accept_optional)
                dpg.add_button(label="Reject", tag="pyecr_optional_reject", enabled=False, callback=self._reject_optional)
            dpg.add_input_text(label="Reviewed outlier original indices (comma list)", tag="pyecr_optional_review_indices", default_value="", width=-1)
            dpg.add_input_text(label="Preview evidence / mask review", tag="pyecr_optional_evidence", default_value="Preview does not change data.", multiline=True, readonly=True, height=TOKENS.heights.history, width=-1)
            dpg.add_input_text(label="Unmask original indices (comma list)", tag="pyecr_optional_unmask_indices", default_value="", width=-1)
            dpg.add_button(label="Unmask reviewed points", tag="pyecr_optional_unmask", callback=self._unmask_optional, enabled=False)
            dpg.add_text("No before/after fit comparison yet.", tag="pyecr_optional_comparison", wrap=TOKENS.wraps.stage)
            with dpg.plot(label="Before / after residuals (check signal scales)", tag='pyecr_optional_residual_plot', height=TOKENS.heights.residual, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Processed time", tag="pyecr_optional_residual_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Observed - fitted", tag="pyecr_optional_residual_axis")
                dpg.add_line_series([], [], label="Before", parent="pyecr_optional_residual_axis", tag="pyecr_optional_before_residual")
                dpg.add_line_series([], [], label="After manual refit", parent="pyecr_optional_residual_axis", tag="pyecr_optional_after_residual")
        self._show_optional_fields("drift_correction")

    def _show_optional_fields(self, kind: str) -> None:
        dpg = self._require_dpg()
        for operation in _EXAMPLES:
            dpg.configure_item(f"pyecr_optional_form_{operation}", show=operation == kind)

    def _typed_optional_options(self, dpg, kind: str) -> dict[str, object]:
        names = {
            "drift_correction": ("drift_method", "reference_start", "reference_end", "scope_start", "scope_end", "anchor_time"),
            "outlier_mask": ("threshold", "half_window", "noise_floor"),
            "adaptive_downsample": ("target_points", "early_until", "late_power"),
        }[kind]
        try:
            values = {name: dpg.get_value(f"pyecr_optional_field_{name}") for name in names}
        except (KeyError, SystemError):
            # Older headless callers supply the previous JSON interface.
            result = json.loads(str(dpg.get_value("pyecr_optional_options")))
            if not isinstance(result, dict):
                raise ValueError("Options must be a JSON object")
            if "parameters" in result:
                result = result["parameters"]
            allowed = {
                "drift_correction": ("method", "reference_window", "scope", "anchor_time"),
                "outlier_mask": ("threshold", "half_window", "noise_floor"),
                "adaptive_downsample": ("target_points", "early_until", "late_power"),
            }[kind]
            return {key: result[key] for key in allowed if key in result}
        return optional_parameters(kind, values)

    def _select_optional_kind(self, sender, app_data, user_data) -> None:
        del sender, user_data
        self._require_dpg().set_value("pyecr_optional_options", json.dumps(_EXAMPLES[str(app_data)]))
        self._reject_optional(None, None, None)
        self._show_optional_fields(str(app_data))

    def _optional_error(self, exc: Exception) -> None:
        self.controller.record_callback_exception(GuiPage.DATA, exc, traceback.format_exc())
        self._optional_input_error=self.controller.state.current_errors.get(GuiPage.DATA)

    def _clear_optional_input_error(self):
        """Resolve this tool's current error without deleting history or other errors."""
        error=getattr(self,'_optional_input_error',None)
        state=self.controller.state
        if error is not None and state.current_errors.get(GuiPage.DATA) is error:
            state.current_errors.pop(GuiPage.DATA,None)
            task=state.tasks[GuiPage.DATA]
            if task.status=='error' and task.message==error.message:
                task.status='idle';task.message=''
        self._optional_input_error=None

    def _preview_optional(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        self._pending_optional_preview = None
        if hasattr(dpg,'does_item_exist'):
            for name in FIELD_DOMAINS:
                tag=f'pyecr_optional_error_{name}'
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag,'');dpg.configure_item(tag,show=False)
        try:
            dataset = self.controller.session.dataset
            if dataset is None:
                raise ValueError("Import data before optional preprocessing")
            kind = str(dpg.get_value("pyecr_optional_kind"))
            options = self._typed_optional_options(dpg, kind)
            factory = {"drift_correction": preview_drift, "outlier_mask": preview_outliers, "adaptive_downsample": preview_downsampling}[kind]
            preview = factory(dataset, **options)
            self._pending_optional_preview = (preview, self.controller.analysis_session.current_revision)
            dpg.set_value("pyecr_optional_options", json.dumps(preview.step.to_dict(), ensure_ascii=False, indent=2))
            dpg.set_value("pyecr_optional_review_indices", ", ".join(str(i) for i in preview.step.parameters.get("raw_indices", ())))
            self._clear_optional_input_error()
        except Exception as exc:
            if hasattr(dpg, "does_item_exist") and dpg.does_item_exist("pyecr_optional_typed_fields"):
                message = str(exc)
                for name in FIELD_DOMAINS:
                    tag = f"pyecr_optional_error_{name}"
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, message if name in message else "")
                        dpg.configure_item(tag,show=name in message)
            self._optional_error(exc)
        self._refresh_widgets()

    def _accept_optional(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        try:
            if getattr(self, '_direct_preprocess_ready', False):
                self._validate_direct_preview('optional')
            pending = getattr(self, "_pending_optional_preview", None)
            if pending is None:
                raise ValueError("Preview before Accept")
            preview, revision = pending
            if revision != self.controller.analysis_session.current_revision:
                raise ValueError("Optional preview is stale; preview again")
            step = self._optional_step_for_accept(preview)
            self.controller.apply_preprocessing(step)
            self._pending_optional_preview = None
        except Exception as exc:
            self._optional_error(exc)
        self._refresh_widgets()

    def _optional_step_for_accept(self, preview):
        """The original reviewed-subset rule, shared by UI preview and Accept."""
        step = preview.accept(self.controller.session.dataset)
        if step.operation == 'outlier_mask':
            from pyecr.preprocessing.optional import _indices
            indices = _indices(parse_indices(self._require_dpg().get_value('pyecr_optional_review_indices')), self.controller.session.dataset.raw_time.size, empty=True)
            p = step.to_dict()['parameters']
            if not set(indices).issubset(p['raw_indices']):
                raise ValueError('Reviewed indices must be a subset of preview suggestions')
            p['raw_indices'] = indices
            p['reasons'] = [row for row in p['reasons'] if row['raw_index'] in indices]
            step = ProcessingStep.create(step.operation, p, timestamp=step.timestamp)
        return step

    def _reject_optional(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        self._pending_optional_preview = None
        self._refresh_widgets()

    def _unmask_optional(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        try:
            indices = parse_indices(self._require_dpg().get_value("pyecr_optional_unmask_indices"))
            pipeline = self.controller.session.preprocessing
            active = set(reviewable_mask_indices(pipeline.config))
            if not indices or not set(indices).issubset(active):
                raise ValueError("Choose currently masked outlier original indices")
            self.controller.apply_preprocessing(ProcessingStep.create("unmask_points", {"raw_indices": indices, "reason": "user reviewed and unmasked"}))
            self._pending_optional_preview = None
        except Exception as exc:
            self._optional_error(exc)
        self._refresh_widgets()

    def _render_optional_preprocessing(self, dpg, dataset) -> None:
        # Small existing render-test doubles do not build optional widgets.
        if not hasattr(dpg, "does_item_exist") or not dpg.does_item_exist("pyecr_optional_controls"):
            return
        pipeline = self.controller.session.preprocessing
        pending = getattr(self, "_pending_optional_preview", None)
        if pending is not None and pending[1] != self.controller.analysis_session.current_revision:
            pending = self._pending_optional_preview = None
        binding=None
        if getattr(self,'_direct_preprocess_ready',False):
            if self._direct_binding_current() and self._direct_preview_binding['kind']=='optional':
                binding=self._direct_preview_binding
            else:pending=None
        active = reviewable_mask_indices(pipeline.config) if pipeline else ()
        evidence = [f"Reviewable masked original indices: {list(active)}", "Unmask replays earlier transforms. Crop/downsample exclusions stay in force."]
        dpg.configure_item("pyecr_optional_preview", enabled=dataset is not None)
        can_accept = bool(pending is not None and (pending[0].step.operation != "outlier_mask" or pending[0].step.parameters["raw_indices"]))
        dpg.configure_item("pyecr_optional_accept", enabled=can_accept)
        dpg.configure_item("pyecr_optional_reject", enabled=pending is not None)
        dpg.configure_item("pyecr_optional_unmask", enabled=bool(active))
        tags=["pyecr_optional_preview_series", "pyecr_optional_markers"]
        if dpg.does_item_exist('pyecr_optional_retained_markers'):tags.append('pyecr_optional_retained_markers')
        for tag in tags:
            dpg.set_value(tag, [[], []])
            dpg.configure_item(tag, show=False)
        if pending is not None and dataset is not None:
            from pyecr.preprocessing.optional import apply_optional
            preview = pending[0]
            evidence.extend(preview.notes)
            step=ProcessingStep.from_dict(binding['step']) if binding else preview.step
            evidence.append(json.dumps(step.to_dict()["parameters"], ensure_ascii=False))
            p = step.parameters
            if preview.step.operation == "outlier_mask":
                selected = np.isin(np.flatnonzero(dataset.mask), p["raw_indices"])
                dpg.set_value("pyecr_optional_markers", [dataset.processed_time[selected].tolist(), dataset.processed_signal[selected].tolist()])
                dpg.configure_item("pyecr_optional_markers", show=True)
            else:
                view = binding['proposed'] if binding else apply_optional(dataset,step)
                dpg.set_value("pyecr_optional_preview_series", [view.processed_time.tolist(), view.processed_signal.tolist()])
                dpg.configure_item("pyecr_optional_preview_series", show=True)
                if step.operation=='adaptive_downsample' and dpg.does_item_exist('pyecr_optional_retained_markers'):
                    dpg.set_value('pyecr_optional_retained_markers',[view.processed_time.tolist(),view.processed_signal.tolist()])
                    dpg.configure_item('pyecr_optional_retained_markers',show=True)
            if preview.step.operation != "outlier_mask" or preview.step.parameters["raw_indices"]:
                candidate = binding['proposed'] if binding else PreprocessConfig((*pipeline.config.steps,step)).replay(pipeline.source)
                evidence.append(preview_evidence(pipeline.source,dataset,candidate,step,pipeline.config))
            else:
                evidence.append("No outlier candidates. Nothing can be accepted; raw and processed data are unchanged.")
            if hasattr(dpg, "fit_axis_data"):
                dpg.fit_axis_data("pyecr_data_processed_y_axis")
        if pipeline:
            for step in pipeline.config.steps:
                if step.operation == "outlier_mask":
                    evidence.append(json.dumps({k: v for k, v in step.to_dict()["parameters"].items() if k != "before_fit"}, ensure_ascii=False))
        dpg.set_value("pyecr_optional_evidence", "\n".join(evidence))
        result = self.controller.analysis_session.fit_result
        current = result if not self.controller.analysis_session.fit_stale else None
        comparison = preprocessing_comparison(dataset, current) if dataset else None
        for side in ("before", "after"):
            summary = comparison.get(side) if comparison else None
            curve = summary.get("residual_curve") if summary else None
            dpg.set_value(f"pyecr_optional_{side}_residual", [[], []] if not curve else [list(curve["time"]), list(curve["residual"])])
        if hasattr(dpg, "fit_axis_data") and comparison:
            dpg.fit_axis_data("pyecr_optional_residual_x_axis")
            dpg.fit_axis_data("pyecr_optional_residual_axis")
        text_comparison = None if comparison is None else {
            "before": {k: v for k, v in comparison["before"].items() if k != "residual_curve"},
            "after": None if comparison["after"] is None else {k: v for k, v in comparison["after"].items() if k != "residual_curve"},
            "note": comparison["note"],
        }
        dpg.set_value("pyecr_optional_comparison", "No before/after fit comparison yet." if comparison is None else json.dumps(text_comparison, default=lambda v: dict(v), ensure_ascii=False))
