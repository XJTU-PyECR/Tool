"""Typed user-triggered P65B scenario selection and read-only comparison view."""
from __future__ import annotations

import json

from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.services.preprocessing_robustness import RobustnessVariation


class PreprocessingRobustnessViewMixin:
    def _build_preprocessing_robustness(self) -> None:
        dpg = self._require_dpg()
        with dpg.group(tag="pyecr_robustness_controls"):
            dpg.add_separator()
            dpg.add_text("P65B · Preprocessing robustness (explicit run)", wrap=TOKENS.wraps.stage)
            dpg.add_text("Assess up to eight independent fits. This does not accept preprocessing or replace the current Fit. Processing-choice sensitivity is not a statistical CI.", wrap=TOKENS.wraps.stage)
            dpg.add_input_text(label="Scenario name", tag="pyecr_robustness_name", default_value="scenario-1", width=-1)
            dpg.add_combo(("initial_window", "final_window", "t0", "drift_reference", "downsample"),
                          label="Perturbation", tag="pyecr_robustness_kind", default_value="initial_window", width=-1)
            dpg.add_input_text(label="Value 1 · start / t0 / target points", tag="pyecr_robustness_value_1", width=-1)
            dpg.add_input_text(label="Value 2 · end / early until", tag="pyecr_robustness_value_2", width=-1)
            dpg.add_input_text(label="Value 3 · late power (downsample)", tag="pyecr_robustness_value_3", default_value="2", width=-1)
            dpg.add_text("Window/reference: start,end in recipe-step time units. t0: replacement value. Downsample: target, early-until, late-power. Existing mask/crop remain.", wrap=TOKENS.wraps.stage)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add scenario", tag="pyecr_robustness_add", callback=self._robustness_add)
                dpg.add_button(label="Remove last", tag="pyecr_robustness_remove", callback=self._robustness_remove)
                dpg.add_button(label="Run assessment", tag="pyecr_robustness_run", callback=self._robustness_run)
                dpg.add_button(label="Cancel", tag="pyecr_robustness_cancel", callback=self._robustness_cancel)
            dpg.add_input_text(label="Selected scenarios", tag="pyecr_robustness_choices", readonly=True,
                               multiline=True, height=TOKENS.heights.history, width=-1)
            dpg.add_text("Requires a current Fit.", tag="pyecr_robustness_status", wrap=TOKENS.wraps.stage)
            dpg.add_input_text(label="Assessment JSON path", tag="pyecr_robustness_path", width=-1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save assessment", tag="pyecr_robustness_save", callback=self._robustness_save)
                dpg.add_button(label="Open assessment", tag="pyecr_robustness_open", callback=self._robustness_open)
            dpg.add_input_text(label="Read-only comparison", tag="pyecr_robustness_summary", readonly=True,
                               multiline=True, height=TOKENS.heights.history, width=-1)

    def _robustness_add(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        try:
            kind = str(dpg.get_value("pyecr_robustness_kind"))
            count = 3 if kind == "downsample" else 1 if kind == "t0" else 2
            values = tuple(float(dpg.get_value(f"pyecr_robustness_value_{i}")) for i in range(1, count + 1))
            choice = RobustnessVariation(str(dpg.get_value("pyecr_robustness_name")), kind, values)
            if len(self._robustness_variations) >= 8 or any(v.name == choice.name for v in self._robustness_variations):
                raise ValueError("Choose at most eight scenarios with distinct names")
            self._robustness_variations.append(choice)
            dpg.set_value("pyecr_robustness_name", f"scenario-{len(self._robustness_variations) + 1}")
            dpg.set_value("pyecr_robustness_status", "Scenario added; Run assessment starts independent fits.")
        except (TypeError, ValueError) as exc:
            dpg.set_value("pyecr_robustness_status", f"Input error: {exc}")
        self._render_robustness(dpg)

    def _robustness_remove(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        if self._robustness_variations:
            self._robustness_variations.pop()
        self._render_robustness(self._require_dpg())

    def _robustness_run(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        try:
            self.controller.run_preprocessing_robustness(tuple(self._robustness_variations))
            self._robustness_last_rendered_terminal = None
            dpg.set_value("pyecr_robustness_status", "Running; the active Fit remains available.")
        except (RuntimeError, ValueError, TypeError) as exc:
            dpg.set_value("pyecr_robustness_status", f"Cannot run: {exc}")
        self._render_robustness(dpg)

    def _robustness_cancel(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        if self.controller.cancel_preprocessing_robustness():
            self._require_dpg().set_value("pyecr_robustness_status", "Cancelling; waiting for worker to exit.")

    def _robustness_save(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        try:
            path = str(dpg.get_value("pyecr_robustness_path")).strip()
            if not path:
                raise ValueError("Enter an assessment JSON path")
            self.controller.save_preprocessing_robustness(path)
            dpg.set_value("pyecr_robustness_status", f"Saved read-only assessment: {path}")
        except (RuntimeError, ValueError, OSError) as exc:
            dpg.set_value("pyecr_robustness_status", f"Save failed: {exc}")

    def _robustness_open(self, sender, app_data, user_data) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        try:
            path = str(dpg.get_value("pyecr_robustness_path")).strip()
            if not path:
                raise ValueError("Enter an assessment JSON path")
            self.controller.open_preprocessing_robustness(path)
            dpg.set_value("pyecr_robustness_status", f"Opened read-only assessment: {path}")
            self._render_robustness(dpg)
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            dpg.set_value("pyecr_robustness_status", f"Open failed: {exc}")

    def _render_robustness(self, dpg) -> None:
        if not hasattr(dpg, "does_item_exist") or not dpg.does_item_exist("pyecr_robustness_controls"):
            return
        dpg.set_value("pyecr_robustness_choices", "\n".join(
            f"{v.name}: {v.kind} {list(v.values)}" for v in self._robustness_variations) or "No scenarios selected.")
        task = self.controller.state.tasks[GuiPage.DIAGNOSTICS]
        running = self.controller._robustness_task_id is not None
        dpg.configure_item("pyecr_robustness_run", enabled=bool(self._robustness_variations) and not running
                           and self.controller.workflow.can_diagnose)
        dpg.configure_item("pyecr_robustness_cancel", enabled=running)
        dpg.configure_item("pyecr_robustness_save", enabled=self.controller.robustness_assessment is not None)
        if running:
            dpg.set_value("pyecr_robustness_status", task.message)
        elif (task.status in {"completed", "warning", "cancelled", "error"}
              and task.message.startswith(("Robustness", "All robustness", "Out of date — robustness"))
              and task.message != self._robustness_last_rendered_terminal):
            dpg.set_value("pyecr_robustness_status", task.message)
            self._robustness_last_rendered_terminal = task.message
        assessment = self.controller.robustness_assessment
        session = self.controller.analysis_session
        render_key = (id(assessment), session.current_revision, session.fit_generation, session.fit_stale)
        if render_key == self._robustness_render_key:
            return
        self._robustness_render_key = render_key
        if assessment is None:
            dpg.set_value("pyecr_robustness_summary", "No completed assessment. Active Fit is unchanged.")
            return
        rows = [{
            "name": s.variation.name, "kind": s.variation.kind, "status": s.status,
            "reason": s.reason, "parent_fit_id": assessment.parent_fit_id,
            "scenario_fit_id": s.fit_id, "fit_revision_id": s.revision_id,
            "seed": assessment.seed, "own_SSE": None if s.fit is None else s.fit.SSE,
            "common_SSE": s.common_sse, "parent_common_SSE": s.parent_common_sse,
            "comparability": s.comparability, "comparison_reason": s.comparable_reason,
            "parameter_changes": s.parameter_changes,
            "residual_structure": s.residual_summary, "early_residuals": s.early_residual_summary,
            "residual_change": s.residual_change, "early_residual_change": s.early_residual_change,
            "identifiability": s.identifiability, "quality_warnings": s.quality_warnings,
            "identifiability_change": s.identifiability_change,
            "warning_stability": s.warning_stability,
            "review_flags": s.review_flags, "affected_original_indices": s.affected_indices,
            "common_original_indices": s.common_original_indices,
            "time_domain": s.time_domain, "parameter_replacement": s.changed_parameters,
        } for s in assessment.scenarios]
        heading = ("Current parent" if self.controller.robustness_current else
                   "Historical assessment · parent no longer current")
        dpg.set_value("pyecr_robustness_summary", heading + "\n" +
                      json.dumps({"parent_SSE_full_eligible": assessment.parent_fit.SSE,
                                  "parent_identifiability": assessment.parent_identifiability,
                                  "parent_residual_structure": assessment.parent_residual_summary,
                                  "parent_early_residuals": assessment.parent_early_residual_summary,
                                  "parent_quality_warnings": assessment.parent_quality_warnings,
                                  "parent_eligible_original_indices": assessment.parent_eligible_original_indices,
                                  "interpretation": "Processing-choice sensitivity, not statistical CI",
                                  "review_flags": assessment.review_flags, "scenarios": rows},
                                 ensure_ascii=False, indent=2))
