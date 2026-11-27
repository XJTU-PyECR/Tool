"""Signed normalization form/review. Only the shared Accept callback commits."""
from __future__ import annotations

import json
from collections.abc import Mapping

from pyecr.gui.signed_normalization import (
    REFERENCE_CHOICES, SIGNED_DISPLAY, SIGNED_FIELDS, SIGNED_OPERATION,
    form_signature, operation_id, prepare_signed_normalization, signed_review_text,
)
from pyecr.gui.theme import TOKENS
from pyecr.preprocessing import applied_signed_normalization, inverse_signed_dct


class SignedNormalizationViewMixin:
    def _build_signed_normalization(self):
        dpg = self._require_dpg()
        with dpg.group(tag="pyecr_signed_controls", show=False):
            dpg.add_text(SIGNED_DISPLAY)
            dpg.add_radio_button(REFERENCE_CHOICES, label="Terminal source", tag="pyecr_signed_reference_source",
                          default_value=REFERENCE_CHOICES[0],
                          callback=self._signed_inputs_changed)
            dpg.add_text("", tag="pyecr_signed_window_status", wrap=TOKENS.wraps.rail)
            with dpg.group(tag="pyecr_signed_external_controls", show=False):
                dpg.add_input_text(label="External sigma_inf", tag="pyecr_signed_external_value",
                                   default_value="", width=TOKENS.widths.field_wide, callback=self._signed_inputs_changed)
                dpg.add_input_text(label="Signal unit (current input)", tag="pyecr_signed_external_unit",
                                   readonly=True, default_value="", width=TOKENS.widths.field_wide)
                dpg.add_checkbox(label="I declare this terminal value known",
                                 tag="pyecr_signed_external_declared", default_value=False, callback=self._signed_inputs_changed)
            dpg.add_input_text(label="Reference source / evidence", tag="pyecr_signed_reference_evidence",
                               default_value="", width=-1, callback=self._signed_inputs_changed)
            dpg.add_button(label="Open Platforms", tag="pyecr_signed_open_platforms",
                           callback=self._choose_preprocess_category, user_data="platforms", show=False)
        with dpg.group(tag="pyecr_signed_review_group", show=False):
            dpg.add_text("", tag="pyecr_signed_compact", wrap=TOKENS.wraps.rail)
            dpg.add_button(label="View Signed-DCT review / overlay", callback=self._show_signed_overlay)
        with dpg.window(label="Signed-DCT normalization · review and overlay", tag="pyecr_signed_review_window",
                        show=False, width=TOKENS.widths.normalization_review, height=TOKENS.heights.normalization_review):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Accept", tag="pyecr_signed_accept", callback=self._accept_typed_preprocessing, enabled=False)
                dpg.add_button(label="Reject", tag="pyecr_signed_reject", callback=self._reject_typed_preprocessing, enabled=False)
                dpg.add_button(label="Close view", callback=lambda: dpg.configure_item("pyecr_signed_review_window", show=False))
            with dpg.child_window(height=TOKENS.heights.results, width=-1, tag="pyecr_signed_review_content"):
                dpg.add_text("Preview before Accept.", tag="pyecr_signed_review_text", wrap=TOKENS.wraps.normalization_review)
            with dpg.plot(label="Signed-DCT overlay · complete eligible observations",
                          tag="pyecr_signed_overlay", height=TOKENS.heights.plot, width=-1, show=False):
                dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                dpg.add_plot_axis(dpg.mvXAxis, label="Processed time", tag="pyecr_signed_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="Input signal", tag="pyecr_signed_input_y")
                dpg.add_plot_axis(dpg.mvYAxis2, label="Signed Y (1)", opposite=True, tag="pyecr_signed_output_y")
                dpg.add_scatter_series([], [], label="Input source · left axis", parent="pyecr_signed_input_y", tag="pyecr_signed_input_series")
                dpg.add_line_series([], [], label="sigma_inf · left axis", parent="pyecr_signed_input_y", tag="pyecr_signed_terminal_series")
                dpg.add_scatter_series([], [], label="Max abs residual · original row", parent="pyecr_signed_input_y", tag="pyecr_signed_max_series")
                for edge in ("start", "end"):
                    dpg.add_line_series([], [], label=f"Accepted final {edge}", parent="pyecr_signed_input_y", tag=f"pyecr_signed_window_{edge}")
                dpg.add_line_series([], [], label="Signed Y · right axis", parent="pyecr_signed_output_y", tag="pyecr_signed_output_series")
        self._register_owned_root("pyecr_signed_review_window")

    def _show_signed_overlay(self, *args):
        dpg = self._require_dpg()
        size = {}
        if hasattr(dpg, "get_viewport_width"):
            size = {"width": min(1000, dpg.get_viewport_width() - 40),
                    "height": min(820, dpg.get_viewport_height() - 60), "pos": (20, 20)}
        dpg.configure_item("pyecr_signed_review_window", show=True, **size)
        dpg.configure_item("pyecr_signed_review_text", wrap=max(240, size.get("width", 1000) - 80))

    def _signed_values(self):
        dpg = self._require_dpg()
        values = {name: dpg.get_value(f"pyecr_signed_{name}") for name in SIGNED_FIELDS}
        if getattr(self, '_direct_preprocess_ready', False):
            values['external_unit'] = dpg.get_value('pyecr_signed_external_unit')
        return values

    def _signed_inputs_changed(self, sender, app_data, user_data):
        if sender == "pyecr_signed_reference_source":
            self._clear_signed_external_declaration()
        self._pending_typed_preview = None
        self._signed_review = None
        self._signed_message = "Inputs changed. Preview again before Accept; data and recipe unchanged."
        self._refresh_widgets()

    def _clear_signed_external_declaration(self):
        dpg = self._require_dpg()
        for name, value in (("external_value", ""), ("reference_evidence", ""), ("external_declared", False)):
            dpg.set_value(f"pyecr_signed_{name}", value)

    def _preview_signed_normalization(self):
        dpg = self._require_dpg()
        self._signed_review = None
        self._pending_typed_preview = None
        try:
            pipeline = self.controller.session.preprocessing
            if pipeline is None:
                raise ValueError("Import a dataset before Signed-DCT normalization")
            values = self._signed_values()
            step, result = prepare_signed_normalization(pipeline.dataset, pipeline.config, values)
            self._signed_review = {"result": result, "dataset": pipeline.dataset, "revision": pipeline.revision,
                                   "data_revision": self.controller.analysis_session.data_revision,
                                   "signature": form_signature(values)}
            if result.state.ready:
                proposed = pipeline.config.append(step).replay(pipeline.source)
                self._pending_typed_preview = (step, proposed, pipeline.dataset, pipeline.revision,
                                               self.controller.analysis_session.data_revision)
            self._signed_message = "Review the overlay, source, units and warnings before Accept." if result.state.ready else "Blocked. Resolve the recorded reason; no step can be accepted."
            dpg.set_value("pyecr_data_processing_parameters", json.dumps(step.to_dict(), ensure_ascii=False, indent=2))
            dpg.set_value("pyecr_typed_error", self._signed_message)
        except Exception as exc:
            self._signed_message = f"Input error: {exc}"
            dpg.set_value("pyecr_typed_error", self._signed_message)
        self._refresh_widgets()
        self._show_signed_overlay()

    def _validate_signed_accept(self):
        review = getattr(self, "_signed_review", None)
        if (review is None or not review["result"].state.ready
                or form_signature(self._signed_values()) != review["signature"]
                or operation_id(self._require_dpg().get_value("pyecr_data_processing_operation")) != SIGNED_OPERATION):
            raise ValueError("Signed preview is unavailable or inputs changed; Preview again before Accept")

    def _render_signed_normalization(self, dpg, pipeline):
        if not hasattr(dpg, "does_item_exist") or not dpg.does_item_exist("pyecr_signed_controls"):
            return
        selected = operation_id(dpg.get_value("pyecr_data_processing_operation")) == SIGNED_OPERATION
        dpg.configure_item("pyecr_signed_controls", show=selected)
        if selected and pipeline is not None:
            binding = (id(pipeline.source), pipeline.dataset.metadata.get("signal_type"), pipeline.dataset.metadata.get("signal_unit"))
            previous = getattr(self, "_signed_form_source", None)
            if previous is not None and previous != binding:
                self._clear_signed_external_declaration()
            self._signed_form_source = binding
        if selected:
            source = self._signed_values()["reference_source"]
            dpg.configure_item("pyecr_signed_external_controls", show=source == REFERENCE_CHOICES[1])
            windows = pipeline.dataset.metadata.get("stability_windows", {}) if pipeline else {}
            window = windows.get("final") if isinstance(windows, Mapping) else None
            if not isinstance(window, Mapping) or not all(k in window for k in ("start_time", "end_time", "time_unit", "n_points")):
                window = None
            description = (f"Accepted Final [{window['start_time']:.9g}, {window['end_time']:.9g}] "
                           f"{window['time_unit']}." if window else
                           "No accepted Final. Open Platforms to Preview / Accept a window.")
            unit = pipeline.dataset.metadata.get("signal_unit") if pipeline else "import data first"
            dpg.set_value("pyecr_signed_window_status", f"Input signal unit: {unit}\n{description}")
            if getattr(self, '_direct_preprocess_ready', False):
                dpg.set_value('pyecr_signed_external_unit', unit)
                dpg.configure_item('pyecr_signed_open_platforms', show=window is None and source == REFERENCE_CHOICES[0])
        review = getattr(self, "_signed_review", None)
        if review is not None and (pipeline is None or review["dataset"] is not pipeline.dataset
                or review["revision"] != pipeline.revision or review["data_revision"] != self.controller.analysis_session.data_revision
                or not selected or review["signature"] != form_signature(self._signed_values())):
            self._signed_review = review = None
            pending = getattr(self, "_pending_typed_preview", None)
            if pending is not None and pending[0].operation == SIGNED_OPERATION:
                self._pending_typed_preview = None
            self._signed_message = "Signed preview expired after a data/input revision. Preview again."
            dpg.configure_item("pyecr_data_accept_processing", enabled=False)
        if selected:
            dpg.configure_item("pyecr_data_reject_processing", enabled=review is not None)
        dpg.configure_item("pyecr_signed_accept", enabled=bool(selected and review and review["result"].state.ready
                           and getattr(self, "_pending_typed_preview", None) is not None))
        dpg.configure_item("pyecr_signed_reject", enabled=bool(selected and review))
        accepted = bool(pipeline and pipeline.config.steps and pipeline.config.steps[-1].operation == SIGNED_OPERATION)
        key = (id(review), id(pipeline.dataset) if pipeline else None,
               id(pipeline.config) if pipeline else None, selected, getattr(self, "_signed_message", ""))
        if key == getattr(self, "_signed_render_key", None):
            return
        self._signed_render_key = key
        result, dataset, input_values = None, None, None
        if review is not None:
            result, dataset = review["result"], review["dataset"]
            input_values = dataset.processed_signal.tolist()
        elif accepted:
            try:
                dataset = pipeline.dataset
                result = applied_signed_normalization(dataset, recipe=pipeline.config)
                input_values = list(inverse_signed_dct(result))
            except (ValueError, TypeError, KeyError) as exc:
                self._signed_message = f"Signed audit is unavailable: {exc}"
        dpg.configure_item("pyecr_signed_review_group", show=selected or accepted)
        message = getattr(self, "_signed_message", "Preview only. Data and recipe unchanged.")
        dpg.set_value("pyecr_signed_review_text", message + ("\n" + signed_review_text(dataset, result) if result else ""))
        compact = message
        if getattr(self, '_direct_preprocess_ready', False):
            # The shared Preview summary owns numbers and warnings. Full row
            # provenance and reversible payload remain available on demand.
            compact = ''
        elif result is not None:
            compact = (f"{result.state.status} | {len(result.spec.raw_indices)} eligible rows\n"
                       + (f"sigma_inf={result.reference_value:.9g}; S={result.scale_value:.9g} {result.spec.signal_unit}\n"
                          f"Max original row index={result.scale_raw_index}; output 1" if result.state.ready else
                          "; ".join(result.state.blocking_reasons))
                       + f"\nWarnings: {len(result.state.warnings)}; open the full review.")
        dpg.set_value("pyecr_signed_compact", compact)
        dpg.configure_item('pyecr_signed_compact', show=bool(compact))
        available = result is not None and result.state.ready
        dpg.configure_item("pyecr_signed_overlay", show=available)
        series = ("input_series", "output_series", "terminal_series", "max_series", "window_start", "window_end")
        if not available:
            for name in series:
                dpg.set_value(f"pyecr_signed_{name}", [[], []])
            return
        times = dataset.processed_time.tolist()
        dpg.set_value("pyecr_signed_input_series", [times, input_values])
        dpg.set_value("pyecr_signed_output_series", [times, list(result.normalized_signal)])
        dpg.set_value("pyecr_signed_terminal_series", [[times[0], times[-1]], [result.reference_value] * 2])
        peak = result.spec.raw_indices.index(result.scale_raw_index)
        dpg.set_value("pyecr_signed_max_series", [[times[peak]], [input_values[peak]]])
        for edge, index in (("start", 0), ("end", -1)):
            if result.spec.reference_rows:
                t = times[result.spec.raw_indices.index(result.spec.reference_rows[index])]
                value = [[t, t], [min(input_values), max(input_values)]]
            else:
                value = [[], []]
            dpg.set_value(f"pyecr_signed_window_{edge}", value)
            dpg.configure_item(f"pyecr_signed_window_{edge}", show=bool(result.spec.reference_rows))
        dpg.configure_item("pyecr_signed_x", label=f"Time ({dataset.metadata['time_unit']})")
        dpg.configure_item("pyecr_signed_input_y", label=f"Input signal ({result.spec.signal_unit})")
        for axis in ("x", "input_y", "output_y"):
            dpg.fit_axis_data(f"pyecr_signed_{axis}")
