"""Pair / Design UI; scientific work lives in independent services."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable

from pyecr.config import BoundaryConfig, SampleGeometry
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.gui.pair_display import PAIR_KIND_LABELS, short_pair_reason
from pyecr.model_display import display_name
from pyecr.services.experiment_design import ExperimentDesignRequest
from pyecr.services.paired_ecr import DIRECTIONS, PAIR_KINDS, _unit_scale


from pyecr.gui.components.pair_layout import PairLayoutMixin


class PairDesignViewMixin(PairLayoutMixin):
    def _build_pair_design_area(self) -> None:
        self._build_pair_layout()

    def _pair_action(self, action: Any, slot: str | None = None, *, success_message: str | Callable[[Any],str] = '') -> bool:
        succeeded = False
        try:
            result = action()
            self._pair_action_error = ''
            if callable(success_message):success_message=success_message(result)
            self._pair_artifact_message = success_message
            owned = getattr(self, '_pair_owned_error', None)
            if owned is not None and self.controller.state.current_errors.get(GuiPage.PAIR) is owned:
                self.controller.state.current_errors.pop(GuiPage.PAIR, None)
                task = self.controller.state.tasks[GuiPage.PAIR]
                if task.status == 'error':
                    task.status = 'idle'; task.message = success_message or 'Action completed.'
            self._pair_owned_error = None
            if slot and hasattr(self, '_pair_card_errors'):
                self._pair_card_errors.pop(slot.lower(), None)
            succeeded = True
        except Exception as exc:
            self._pair_action_error = '' if slot else short_pair_reason(exc)
            if slot and hasattr(self, '_pair_card_errors'):
                self._pair_card_errors[slot.lower()] = short_pair_reason(exc)
            self.controller.record_callback_exception(GuiPage.PAIR, exc, traceback.format_exc())
            self._pair_owned_error = self.controller.state.current_errors.get(GuiPage.PAIR)
            self._pair_artifact_message = ''
        self._refresh_widgets()
        return succeeded

    def _pair_input(self, slot: str | None = None, *, fallback: str = '') -> dict[str, object]:
        dpg = self._require_dpg()
        suffix = '' if slot is None else '_' + slot.lower()
        context = {}
        if dpg.get_value("pyecr_pair_record_temperature" + suffix):
            context["temperature_k"] = float(dpg.get_value("pyecr_pair_temperature" + suffix))
        for phase in ("initial", "final"):
            text = str(dpg.get_value(f"pyecr_pair_{phase}" + suffix)).strip()
            if text:
                context[phase + "_condition"] = text
        data = self.controller.analysis_session.dataset
        source_name = '' if data is None else str(data.metadata.get('dataset_name', ''))
        return {"label": str(dpg.get_value("pyecr_pair_label" + suffix)).strip() or fallback or source_name or self.controller.state.dataset_name or "ECR curve",
                "direction": str(dpg.get_value("pyecr_pair_direction" + suffix)), "context": context}

    def _pair_capture(self, sender=None, app_data=None, user_data=None) -> None:
        slot = user_data or str(self._require_dpg().get_value("pyecr_pair_slot"))
        self._pair_action(lambda: self.controller.capture_pair_member(slot, **self._pair_input(user_data)), slot)

    def _pair_load(self, *args: Any) -> None:
        dpg = self._require_dpg()
        self._pair_action(lambda: self.controller.load_pair_project(str(dpg.get_value("pyecr_pair_slot")), str(dpg.get_value("pyecr_pair_project_path")).strip(), **self._pair_input()))

    def _show_pair_load(self, sender=None, app_data=None, user_data=None) -> None:
        d = self._require_dpg(); self._pair_load_slot = user_data
        d.set_value('pyecr_pair_load_title', 'Source ' + user_data + ' — direction and declarations apply when Load is confirmed.')
        d.set_value('pyecr_pair_load_path', ''); d.set_value('pyecr_pair_load_reason', '')
        self._size_display_window('pyecr_pair_load_window'); d.show_item('pyecr_pair_load_window')

    def _pair_load_selected(self, *args: Any) -> None:
        d = self._require_dpg(); slot = self._pair_load_slot
        path = str(d.get_value('pyecr_pair_load_path')).strip()
        def load():
            if not path: raise ValueError('Enter the complete project directory.')
            self.controller.load_pair_project(slot, path, **self._pair_input(slot, fallback=Path(path).name))
        if self._pair_action(load, slot):
            d.hide_item('pyecr_pair_load_window')
        else:
            d.set_value('pyecr_pair_load_reason', self._pair_card_errors.get(slot.lower(), 'Review the project path and captured conditions.'))

    def _pair_compare(self, *args: Any) -> None:
        selected = str(self._require_dpg().get_value("pyecr_pair_kind"))
        self._pair_action(lambda: self.controller.compare_pair(PAIR_KIND_LABELS.get(selected, selected)))

    def _design_request(self) -> ExperimentDesignRequest:
        dpg = self._require_dpg()
        mode = str(dpg.get_value("pyecr_design_boundary"))
        if mode == "exponential_flush":
            boundary = BoundaryConfig(mode=mode, flush_time_s=float(dpg.get_value("pyecr_design_flush")))
        elif mode == "measured_history":
            boundary = BoundaryConfig(mode=mode,
                measured_time_s=tuple(json.loads(dpg.get_value("pyecr_design_boundary_time"))),
                measured_values=tuple(json.loads(dpg.get_value("pyecr_design_boundary_values"))),
                measured_signal_type=str(dpg.get_value("pyecr_design_gas")),
                measured_initial_value=float(dpg.get_value("pyecr_design_initial_gas")),
                measured_final_value=float(dpg.get_value("pyecr_design_final_gas")),
                measured_value_unit=str(dpg.get_value("pyecr_design_gas_unit")))
        else:
            boundary = BoundaryConfig(mode=mode)
        active = tuple(bool(dpg.get_value(f"pyecr_design_active_{axis}")) for axis in "xyz")
        return ExperimentDesignRequest(
            tuple(json.loads(dpg.get_value("pyecr_design_d"))), tuple(json.loads(dpg.get_value("pyecr_design_k"))),
            tuple(SampleGeometry(tuple(dimensions), active) for dimensions in json.loads(dpg.get_value("pyecr_design_geometries"))),
            tuple(json.loads(dpg.get_value("pyecr_design_durations"))), float(dpg.get_value("pyecr_design_noise")),
            int(dpg.get_value("pyecr_design_points")), str(dpg.get_value("pyecr_design_sampling")),
            int(dpg.get_value("pyecr_design_range_samples")), boundary)

    def _design_run(self, *args: Any) -> None:
        def run():
            reason = self._design_draft_reason()
            if reason:raise ValueError(reason)
            self.controller.run_experiment_design(self._design_request())
        self._pair_action(run)

    def _design_cancel(self, *args: Any) -> None:
        self._pair_action(lambda: self.controller.cancel_page(GuiPage.PAIR))

    def _snapshot_path(self) -> Path:
        text = str(self._require_dpg().get_value("pyecr_pair_design_path")).strip()
        if not text:
            raise ValueError("Enter a snapshot path before saving, opening or exporting")
        return Path(text)

    def _pair_design_save(self, *args: Any) -> None:
        self._pair_action(lambda: self.controller.save_pair_design_snapshot(self._snapshot_path()),success_message=lambda path:'Snapshot saved: '+str(path))

    def _pair_design_open(self, *args: Any) -> None:
        self._pair_action(lambda: self.controller.open_pair_design_snapshot(self._snapshot_path()),success_message='Snapshot opened. Stored results restored; Analysis is unchanged.')

    def _pair_design_report(self, *args: Any) -> None:
        self._pair_action(lambda: self.controller.session.pair_design_state.export_report(self._snapshot_path().with_suffix(".html")),success_message=lambda path:'HTML exported: '+str(path))

    def _pair_design_clear(self, *args: Any) -> None:
        self._pair_action(self.controller.clear_pair_design)

    def _refresh_pair_design_widgets(self) -> None:
        self._refresh_pair_layout()
        dpg = self._require_dpg()
        state = self.controller.session.pair_design_state
        running = self.controller.state.tasks[GuiPage.PAIR].status == "running"
        dpg.configure_item("pyecr_pair_capture", enabled=self.controller.can_capture_pair_member and not running)
        if hasattr(self, '_pair_card_members'):
            for slot in ('a', 'b'):
                dpg.configure_item('pyecr_pair_capture_' + slot, enabled=self.controller.can_capture_pair_member and not running)
                dpg.configure_item('pyecr_pair_load_' + slot, enabled=not running)
            dpg.configure_item('pyecr_pair_load_confirm', enabled=not running)
        dpg.configure_item("pyecr_pair_compare", enabled=state.member_a is not None and state.member_b is not None and not running)
        for tag in ("pyecr_pair_load", "pyecr_design_run", "pyecr_pair_design_save", "pyecr_pair_design_open", "pyecr_pair_design_clear", "pyecr_pair_design_report"):
            dpg.configure_item(tag, enabled=not running)
        if hasattr(self,'_draft_geometry_tags'):
            dpg.configure_item('pyecr_design_run',enabled=not running and not self._design_draft_reason())
        dpg.configure_item("pyecr_design_cancel", enabled=running)
        if state is self._rendered_pair_design_state:
            return
        self._rendered_pair_design_state = state
        dpg.set_value("pyecr_pair_members", "\n".join(
            f"{slot}: {m.label} · {m.direction} · {display_name(m.snapshot.analysis_config.model_id)} · Fit {m.fit_id[:12]}"
            if m is not None else f"{slot}: not captured" for slot, m in (("A", state.member_a), ("B", state.member_b))))
        result = state.comparison
        lines = ["No comparison. Review directions and select Check and Compare."]
        if result is not None:
            lines = [f"PAIR READY {'YES' if result.ready else 'NO'}", "Blocking: " + "; ".join(result.compatibility.blocking_reasons or ("none",)),
                     "Warnings: " + "; ".join(result.compatibility.warnings or ("none",))]
            lines.extend(f"{row.parameter}: A={row.value_a:.5g}, B={row.value_b:.5g}; Δ={row.difference_b_minus_a:.5g}; relative={row.relative_difference:.2%}; symmetric={row.symmetric_relative_difference:.2%}" for row in result.parameters)
            for c in result.curves:
                lines.extend((f"{c.label}: SSE={c.SSE_reference:.5g}; RMSE={c.RMSE_reference:.5g}; R²={c.R2:.5g}; residual mean/SD/max={c.residual_mean_reference:.4g}/{c.residual_sd_reference:.4g}/{c.residual_max_abs_reference:.4g} ({c.residual_unit}); fit ready={c.fit_ready}",
                              f"Identifiability: {c.identifiability}; {c.identifiability_explanation}", f"Model adequacy: {c.model_adequacy}; {c.model_adequacy_explanation}"))
            early = result.early_time_mismatch
            lines.append(f"Early mismatch: {early.reason}; fit RMS={early.rms_fit_difference}; observed RMS={early.rms_observed_difference}")
            lines.extend(result.interpretation_notes)
        dpg.set_value("pyecr_pair_result", "\n".join(lines))
        for suffix, member in (("a", state.member_a), ("b", state.member_b)):
            fit_data = residual_data = [[], []]
            if result is not None and result.ready and member is not None:
                factor, unit = _unit_scale(member.snapshot)
                fit_data = [list(member.snapshot.dataset.processed_time), list(member.snapshot.fit_result.y_fit * factor)]
                residual_data = [list(member.snapshot.dataset.processed_time), list(member.snapshot.fit_result.residuals * factor)]
                from pyecr.plotting.labels import signal_unit_axis_label
                dpg.configure_item("pyecr_pair_fit_y", label=signal_unit_axis_label(unit))
                dpg.configure_item("pyecr_pair_residual_y", label=f"Residual ({unit})")
            dpg.set_value(f"pyecr_pair_fit_{suffix}", fit_data)
            dpg.set_value(f"pyecr_pair_residual_{suffix}", residual_data)
        for axis in ("pyecr_pair_fit_x", "pyecr_pair_fit_y", "pyecr_pair_residual_x", "pyecr_pair_residual_y"):
            dpg.fit_axis_data(axis)
        plans = state.plans
        lines = [f"Saved planning snapshots: {len(plans)}"]
        if plans:
            plan = plans[-1]
            lines.append("Suggested region: " + (", ".join(plan.recommended_candidate_ids) or "none under sampled assumptions"))
            for region in plan.recommended_regions:
                lines.append(f"Suggested intervals: full dimensions (cm)={region.full_dimension_ranges_cm}; active={region.active_dimensions}; duration (s)={region.duration_range_s}. {region.explanation}")
            for c in plan.candidates:
                lines.extend((f"{c.candidate_id}: full cm={c.geometry.full_dims_cm}; active={c.geometry.active_dimensions}; duration={c.duration_s:g}s; Bi ranges={c.bi_ranges}",
                              f"Peak S_logD range={c.D_peak_sensitivity_range}; S_logk range={c.k_peak_sensitivity_range}; conditional separation={c.supported_fraction:.0%}; recommended={c.recommended_region}",
                              "; ".join(c.reasons) or "No screening concerns on sampled scenarios"))
            lines.extend(plan.notes)
            ids = [c.candidate_id for c in plan.candidates]
            selected = dpg.get_value('pyecr_design_candidate')
            dpg.configure_item("pyecr_design_candidate", items=ids)
            dpg.set_value("pyecr_design_candidate", selected if selected in ids else ids[0])
        else:
            dpg.configure_item("pyecr_design_candidate", items=[])
            dpg.set_value("pyecr_design_candidate", "")
        dpg.set_value("pyecr_design_result", "\n".join(lines))
        self._design_preview()

    def _design_preview(self, *args: Any) -> None:
        dpg = self._require_dpg()
        plans = self.controller.session.pair_design_state.plans
        text = "No planning preview."
        for tag in ("pyecr_design_sensitivity_d", "pyecr_design_sensitivity_k"):
            dpg.set_value(tag, [[], []])
        if plans:
            candidate = next((c for c in plans[-1].candidates if c.candidate_id == dpg.get_value("pyecr_design_candidate")), None)
            index = int(dpg.get_value("pyecr_design_scenario"))
            if candidate is not None and 0 <= index < len(candidate.scenarios):
                s = candidate.scenarios[index]
                text = f"D={s.D_cm2_s:g}, k={s.k_cm_s:g}; rank={s.jacobian_rank}; condition={s.condition_number}; correlation={s.correlation}; nominal log SD D/k={s.nominal_logD_sd}/{s.nominal_logk_sd}; response={s.response_fraction}; " + "; ".join(s.reasons)
                for tag, curve in zip(("pyecr_design_sensitivity_d", "pyecr_design_sensitivity_k"), s.sensitivity.curves):
                    dpg.set_value(tag, [list(candidate.time_s), list(curve.values)])
            else:
                text = "Choose a valid candidate and scenario index."
        dpg.set_value("pyecr_design_preview_summary", text)
        dpg.fit_axis_data("pyecr_design_sensitivity_x")
        dpg.fit_axis_data("pyecr_design_sensitivity_y")
