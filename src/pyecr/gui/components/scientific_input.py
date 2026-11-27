"""Reusable scientific input widgets and display helpers."""

from __future__ import annotations
from pyecr.gui.theme import TOKENS

from typing import Any
from pyecr.gui.components.tags import WidgetTags


def _model_parameter_tag(parameter_name: str, field_name: str) -> str:
    return WidgetTags.model_parameter(parameter_name, field_name)


class ScientificInputMixin:
    def _render_model_controls(self, model_id: str) -> None:
        dpg = self._require_dpg()
        from pyecr.gui.model_display import model_id_for_display,display_label
        model_id=model_id_for_display(model_id)
        if getattr(self,'_parameter_window_open',False):self._close_parameter_table()
        # Pinned controls are outside the schema scroll child after R05.
        for tag in ('pyecr_model_apply','pyecr_model_apply_status','pyecr_parameter_table_expand','pyecr_parameter_copy_draft','pyecr_model_assumptions','pyecr_model_limitations'):
            if dpg.does_item_exist(tag):
                item=dpg.get_alias_id(tag)
                wrapper=self._owned_form_fields.get(item,(item,))[0]
                dpg.delete_item(wrapper)
        dpg.set_value('pyecr_model_id',model_id)
        dpg.set_value('pyecr_model_display',display_label(model_id))
        descriptor = self.controller.model_descriptor(model_id)
        defaults = self.controller.default_model_form(descriptor.model_id)
        current = self.controller.current_analysis_config
        portfolio = (current.model_specific.get("local_solver", {})
                     if current is not None and current.model_id.value == model_id else {})
        if dpg.does_item_exist('pyecr_stage_boundary'):
            dpg.move_item('pyecr_stage_boundary',parent='pyecr_stage_model_content')
        if dpg.does_item_exist("pyecr_boundary_controls"):
            dpg.delete_item("pyecr_boundary_controls")
        if dpg.get_item_children("pyecr_model_schema_container", 1):
            dpg.delete_item("pyecr_model_schema_container", children_only=True)
        with dpg.group(parent="pyecr_model_schema_container"):
            from pyecr.model_display import display_name
            dpg.add_text(display_name(descriptor.model_id))
            with dpg.collapsing_header(label='Model interpretation / assumptions',default_open=False):
                dpg.add_text(descriptor.description, wrap=TOKENS.wraps.rail)
            if model_id == "M7_SINGLE_K":
                dpg.add_text("Analytic surface-control k-only · ideal_step. Bi N/A without D; surface-exchange and diffusion-mode times differ.", wrap=TOKENS.wraps.rail)
            if model_id=="M8_DUAL_K":
                dpg.add_text("Analytic dual k · whole-time joint fit; effective processes, not assigned carriers. ideal_step only.",wrap=TOKENS.wraps.rail)
            dpg.add_separator()
            self._build_geometry_table(descriptor,defaults)
            dpg.add_text('',tag='pyecr_model_experiment_summary')
            with dpg.collapsing_header(label='Experiment / boundary — review conditions',tag='pyecr_model_experiment_inputs',default_open=False):
                dpg.add_combo(
                    descriptor.supported_surface_modes,
                    label="Surface mode",
                    tag="pyecr_model_surface_mode",
                    default_value=defaults.surface_mode,
                    width=TOKENS.widths.input_medium,
                )
                dpg.add_combo(
                    descriptor.supported_step_types,
                    label="Step type",
                    tag="pyecr_model_step_type",
                    default_value=defaults.step_type,
                    callback=self._select_model_step_type,
                    width=TOKENS.widths.input_medium,
                )
            with dpg.group(tag="pyecr_boundary_controls"):
                dpg.add_combo(
                    descriptor.supported_boundary_modes,
                    label="Boundary mode",
                    tag="pyecr_model_boundary_mode",
                    default_value=defaults.boundary_mode,
                    callback=self._select_boundary_mode,
                    width=TOKENS.widths.input_wide,
                )
                dpg.add_input_double(
                    label="Flush time (s)",
                    tag="pyecr_boundary_flush_time_s",
                    default_value=1.0,
                    show=False,
                    format="%.3e",
                )
                dpg.add_text(
                    "Flush assessment: run a current single D+k Fit to compare timescales.",
                    tag="pyecr_boundary_flush_assessment",
                    wrap=TOKENS.wraps.rail,
                )
                with dpg.group(tag="pyecr_boundary_measured_group", show=False):
                    dpg.add_input_text(
                        label="Measured time (s)",
                        tag="pyecr_boundary_measured_time_s",
                        default_value="",
                        width=-1,
                    )
                    dpg.add_input_text(
                        label="Measured values",
                        tag="pyecr_boundary_measured_values",
                        default_value="",
                        width=-1,
                    )
                    dpg.add_combo(
                        ("pO2", "pH2O"),
                        label="Measured signal type",
                        tag="pyecr_boundary_measured_signal_type",
                        default_value="pO2",
                    )
                    dpg.add_input_double(
                        label="Initial boundary value",
                        tag="pyecr_boundary_measured_initial",
                        default_value=0.0,
                        format="%.3e",
                    )
                    dpg.add_input_double(
                        label="Final boundary value",
                        tag="pyecr_boundary_measured_final",
                        default_value=1.0,
                        format="%.3e",
                    )
                    dpg.add_input_text(
                        label="Measured value unit",
                        tag="pyecr_boundary_measured_unit",
                        default_value="unspecified",
                    )
                    dpg.add_input_double(
                        label="Resample time step (s)",
                        tag="pyecr_boundary_resample_step_s",
                        default_value=0.1,
                        format="%.3e",
                    )
                    dpg.add_input_int(
                        label="Maximum internal points",
                        tag="pyecr_boundary_max_internal_points",
                        default_value=200000,
                    )
            with dpg.group(tag="pyecr_model_sign_group", show=defaults.step_type == 'custom'):
                dpg.add_input_double(
                    label="Input process 1 sign" if model_id=="M8_DUAL_K" else "Custom H sign",
                    tag="pyecr_model_sign_h",
                    default_value=-1.0,
                    enabled=defaults.step_type == "custom",
                    width=TOKENS.widths.input_medium,
                    format="%.3e",
                )
                dpg.add_input_double(
                    label="Input process 2 sign" if model_id=="M8_DUAL_K" else "Custom O sign",
                    tag="pyecr_model_sign_o",
                    default_value=1.0,
                    enabled=defaults.step_type == "custom",
                    width=TOKENS.widths.input_medium,
                    format="%.3e",
                )
            self._build_parameter_table(descriptor,defaults)
            with dpg.collapsing_header(label='Derived results — read only',tag='pyecr_model_derived_header',default_open=False):
                dpg.add_input_text(tag='pyecr_model_derived_values',readonly=True,multiline=True,width=-1,height=TOKENS.analysis.derived_height)
            with dpg.collapsing_header(label="Numerical and fit controls (Expert)", tag="pyecr_model_numerical_controls", default_open=False):
                dpg.add_input_double(
                    label="Standard optimizer tolerance" if model_id in {"M7_SINGLE_K","M8_DUAL_K"} else "Kernel relative tolerance",
                    tag="pyecr_model_relative_tolerance",
                    default_value=defaults.relative_tolerance,
                    width=TOKENS.widths.input_narrow,
                    format="%.3e",
                )
                if model_id not in {"M7_SINGLE_K","M8_DUAL_K"}:
                    dpg.add_input_double(
                        label="Kernel absolute tolerance",
                        tag="pyecr_model_absolute_tolerance",
                        default_value=defaults.absolute_tolerance,
                        width=TOKENS.widths.input_narrow,
                        format="%.3e",
                    )
                    dpg.add_input_int(
                        label="Maximum series terms",
                        tag="pyecr_model_max_terms",
                        default_value=defaults.max_terms,
                        width=TOKENS.widths.input_medium,
                    )
                dpg.add_input_int(
                    label="Maximum fit evaluations",
                    tag="pyecr_model_max_nfev",
                    default_value=defaults.max_nfev,
                    width=TOKENS.widths.input_regular,
                )
                dpg.add_input_text(label='Random seed (blank = automatic)',tag='pyecr_model_random_seed',default_value='',width=-1)
                dpg.add_text('Apply seed edits. Run allocates and records the actual seed before any initialization; existing engine defaults are preserved.',wrap=TOKENS.wraps.rail)
                dpg.add_combo(
                    ("linear",) if model_id in {"M7_SINGLE_K","M8_DUAL_K"} else ("linear", "soft_l1", "huber", "cauchy", "arctan"),
                    label="Loss",
                    tag="pyecr_model_loss",
                    default_value=defaults.loss,
                    width=TOKENS.widths.input_small,
                )
                dpg.add_combo(("trf",) if model_id in {"M7_SINGLE_K","M8_DUAL_K"} else ("trf", "dogbox", "lm_transformed_experimental"),
                    label="Local solver", tag="pyecr_model_local_solver", default_value=portfolio.get("strategy", "trf"), width=-1)
                if model_id not in {"M7_SINGLE_K","M8_DUAL_K"}:
                    dpg.add_combo(("warn", "reject"), label="Dogbox rank/condition policy",
                        tag="pyecr_model_dogbox_policy", default_value=portfolio.get("dogbox_policy", "warn"), width=-1)
                dpg.add_text("TRF remains default. Dogbox is not recommended for deficient/high-condition Jacobians. LM is experimental and linear-loss only; no automatic solver fallback.", wrap=TOKENS.wraps.rail)
            dpg.add_input_text(
                label="Assumptions",
                tag="pyecr_model_assumptions",
                default_value="\n".join(f"• {item}" for item in descriptor.assumptions),
                multiline=True,
                readonly=True,
                width=-1,
                height=TOKENS.heights.notes,
            )
            if model_id in {"M1_SINGLE_DK","M3_DUAL_HO_DK"}:
                record=current.model_specific.get("fixed_d",{}) if current is not None and current.model_id.value==model_id else {}
                with dpg.collapsing_header(label="Fixed D / conditional k — finite-D model; separate from analytic 02/05",tag='pyecr_fixed_d_conditions',default_open=bool(record)):
                    dpg.add_checkbox(label="Enable explicit fixed-D source entry",tag="pyecr_fixed_d_enabled",default_value=bool(record))
                    dpg.add_text("D values are exact conditions. Review observation free/fixed choices; source uncertainty is not propagated.",wrap=TOKENS.wraps.rail)
                    for name in (("D_cm2_s",) if model_id=="M1_SINGLE_DK" else ("D_H_cm2_s","D_O_cm2_s")):
                        source=record.get("sources",{}).get(name,{})
                        dpg.add_text(f"{name}: use its parameter value above (cm²/s)")
                        dpg.add_combo(("manual","external_measurement","literature","parent_fit"),label="Value source",tag="pyecr_fixed_d_kind_"+name,
                                      default_value=source.get("kind","manual"),width=-1)
                        dpg.add_input_text(label="Reference / protocol",tag="pyecr_fixed_d_reference_"+name,default_value=source.get("reference",""),width=-1)
                        dpg.add_input_text(label="Uncertainty description",tag="pyecr_fixed_d_uncertainty_"+name,
                                           default_value=source.get("uncertainty","unknown; not propagated"),width=-1)
            dpg.add_input_text(
                label="Limitations",
                tag="pyecr_model_limitations",
                default_value="\n".join(f"• {item}" for item in descriptor.limitations),
                multiline=True,
                readonly=True,
                width=-1,
                height=TOKENS.heights.notes,
            )
            dpg.add_button(
                label="Apply",
                tag="pyecr_model_apply",
                callback=self._apply_model_configuration,
                **({'parent':'pyecr_model_action_bar'} if dpg.does_item_exist('pyecr_model_action_bar') else {}),
            )
            dpg.add_text(
                "No model configuration has been applied in this session.",
                tag="pyecr_model_apply_status",
                wrap=TOKENS.wraps.page,
            )
        if dpg.does_item_exist("pyecr_stage_boundary_content"):
            dpg.move_item("pyecr_boundary_controls", parent="pyecr_stage_boundary_content")
            dpg.move_item('pyecr_stage_boundary',parent='pyecr_model_experiment_inputs')
        self._prepare_owned_form('pyecr_model_schema_container')
        self._prepare_owned_form('pyecr_boundary_controls')
        for tag in ('pyecr_model_experiment_inputs','pyecr_boundary_measured_group','pyecr_model_sign_group','pyecr_model_numerical_controls'):
            self._pair_owned_fields(tag)
        self._refresh_owned_analysis_forms()
        self._move_model_evidence()
        if dpg.does_item_exist('pyecr_model_action_bar'):
            dpg.move_item('pyecr_model_apply',parent='pyecr_model_action_bar',before='pyecr_parameter_table_expand')


def _parse_numeric_sequence(value: object, name: str) -> tuple[float, ...]:
    """Parse GUI text only; boundary resampling/convolution stays in core."""

    tokens = str(value).replace(",", " ").split()
    if not tokens:
        raise ValueError(f"{name} must contain at least one value")
    try:
        return tuple(float(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-numeric value") from exc


def _file_dialog_path(app_data: object) -> str:
    if not isinstance(app_data, dict):
        raise ValueError("file dialog did not return a selected path")
    direct = app_data.get("file_path_name")
    if isinstance(direct, str) and direct.strip():
        return direct
    selections = app_data.get("selections")
    if isinstance(selections, dict):
        paths = [value for value in selections.values() if isinstance(value, str)]
        if len(paths) == 1 and paths[0].strip():
            return paths[0]
    raise ValueError("select exactly one CSV, TXT, or XLSX file")


def _directory_dialog_path(app_data: object) -> str:
    if not isinstance(app_data, dict):
        raise ValueError("directory dialog did not return a selected path")
    direct = app_data.get("file_path_name")
    if isinstance(direct, str) and direct.strip():
        return direct
    current = app_data.get("current_path")
    if isinstance(current, str) and current.strip():
        return current
    selections = app_data.get("selections")
    if isinstance(selections, dict):
        paths = [value for value in selections.values() if isinstance(value, str)]
        if len(paths) == 1 and paths[0].strip():
            return paths[0]
    raise ValueError("select exactly one project directory")


def _project_directory_text(value: object, *, require_absolute: bool = False) -> str:
    """Validate a path without rewriting user text or guessing dialog mistakes."""
    from pathlib import Path, PureWindowsPath
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Enter a project directory.')
    windows=PureWindowsPath(value)
    # DPG's directory field can prefix its current directory to an absolute
    # input. Refuse the malformed two-drive result; never reconstruct a path.
    remainder=value[len(windows.drive):]
    if windows.drive and ':' in remainder:
        raise ValueError('The directory selector returned an invalid path. Use Open by full path instead.')
    if require_absolute and not (Path(value).is_absolute() or windows.is_absolute()):
        raise ValueError('Enter a full absolute project path, including the drive and directory.')
    return value
