"""Reusable results table widgets and display helpers."""

from __future__ import annotations

from typing import Any
from pyecr.gui.components.widgets import table_style_kwargs
from pyecr.gui.theme import TOKENS


def format_scientific(value: float) -> str:
    """Uniform display formatting; never alters stored scientific values."""

    return f"{value:.4e}"

def _replace_diagnostics_table(
    dpg: Any,
    *,
    table_tag: str,
    parent: str,
    headings: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> None:
    """Replace one dynamic Diagnostics table without retaining child aliases."""

    if dpg.does_item_exist(table_tag):
        dpg.delete_item(table_tag)
    table = dpg.add_table(
        parent=parent,
        tag=table_tag,
        **table_style_kwargs(),
        policy=getattr(dpg, "mvTable_SizingFixedFit", dpg.mvTable_SizingStretchProp),
        scrollX=True, scrollY=True, width=-1,
        height=TOKENS.heights.diagnostics,
        inner_width=max(400, len(headings)*150),
    )
    for heading in headings:
        dpg.add_table_column(label=heading, parent=table, width_fixed=True, init_width_or_weight=150)
    for values in rows:
        row = dpg.add_table_row(parent=table)
        for value in values:
            dpg.add_text(value, parent=row)
    # Full reasons, CI and sources can be selected/copied, independently of
    # column clipping. Keep one stable sibling; deleting a table does not leak
    # copies or aliases. No numeric values are transformed here.
    if hasattr(dpg, "add_input_text"):
        copy_tag = f"{table_tag}_full_text"
        text = "\t".join(headings) + "\n" + "\n".join("\t".join(values) for values in rows)
        if dpg.does_item_exist(copy_tag):
            dpg.set_value(copy_tag, text)
        else:
            dpg.add_input_text(label="Full table text (select / copy)", tag=copy_tag,
                               parent=parent, default_value=text, readonly=True,
                               multiline=True, width=-1, height=TOKENS.heights.notes)


def _render_bi_table(dpg: Any, presentation: Any) -> None:
    section = presentation.bi
    if not section.available:
        dpg.set_value("pyecr_diagnostics_bi_status", f"Bi: N/A — {section.reason}")
        if dpg.does_item_exist("pyecr_diagnostics_bi_table"):
            dpg.delete_item("pyecr_diagnostics_bi_table")
        _clear_table_copy(dpg, "pyecr_diagnostics_bi_table", section.reason)
        return
    dpg.set_value("pyecr_diagnostics_bi_status", "Bi per active dimension")
    _replace_diagnostics_table(
        dpg,
        table_tag="pyecr_diagnostics_bi_table",
        parent="pyecr_diagnostics_bi_container",
        headings=("Process", "Axis", "Full dimension (cm)", "Bi", "Regime"),
        rows=tuple(
            (
                row.process,
                row.axis,
                format_scientific(row.full_dimension_cm),
                format_scientific(row.value),
                row.regime,
            )
            for row in section.rows
        ),
    )


def _render_parameter_diagnostics_table(dpg: Any, presentation: Any) -> None:
    section = presentation.parameter_uncertainty
    evaluation = getattr(presentation, "evaluation", None)
    dpg.set_value(
        "pyecr_diagnostics_parameter_status",
        (f"Diagnostic local 95% intervals; {evaluation.interval_label}. "
         "Each row evaluates its own precision; error assumptions are separate."
         if evaluation is not None else "Local 95% CI. Numerical contact differs from credibility proximity; full reasons below.")
        if section.available
        else f"Parameter uncertainty: N/A — {section.reason}",
    )
    _replace_diagnostics_table(
        dpg,
        table_tag="pyecr_diagnostics_parameter_table",
        parent="pyecr_diagnostics_parameter_container",
        headings=(
            "Parameter",
            "Estimate",
            "Unit",
            "95% CI",
            "Numerical contact",
            "Local precision / reason" if evaluation is not None else "Credibility / reason",
        ),
        rows=tuple(
            (
                row.name,
                format_scientific(row.estimate),
                row.unit,
                (
                    _format_confidence_interval(row.ci_lower, row.ci_upper)
                    if row.ci_lower is not None and row.ci_upper is not None
                    else f"N/A — {row.ci_reason}"
                ),
                (
                    f"yes ({row.bound_side}) — {row.bound_reason or 'tight contact rule'}"
                    if row.bound_hit
                    else f"no — {row.bound_reason or 'tight contact rule; inspect gate separately'}"
                    if row.bound_hit is False
                    else f"N/A — {row.bound_reason}"
                ),
                "yes" if row.reliable else f"no — {row.reliability_reason}",
            )
            for row in presentation.parameters
        ),
    )


def _render_correlation_table(dpg: Any, presentation: Any) -> None:
    section = presentation.correlation
    if not section.available:
        dpg.set_value(
            "pyecr_diagnostics_correlation_status",
            f"Correlation matrix: N/A — {section.reason}",
        )
        if dpg.does_item_exist("pyecr_diagnostics_correlation_table"):
            dpg.delete_item("pyecr_diagnostics_correlation_table")
        _clear_table_copy(dpg, "pyecr_diagnostics_correlation_table", section.reason)
        return
    dpg.set_value(
        "pyecr_diagnostics_correlation_status",
        "Parameter correlation matrix",
    )
    _replace_diagnostics_table(
        dpg,
        table_tag="pyecr_diagnostics_correlation_table",
        parent="pyecr_diagnostics_correlation_container",
        headings=("Parameter", *section.parameter_names),
        rows=tuple(
            (
                name,
                *(
                    "N/A" if value is None else format_scientific(value)
                    for value in matrix_row
                ),
            )
            for name, matrix_row in zip(
                section.parameter_names,
                section.matrix,
                strict=True,
            )
        ),
    )


def _render_multistart_table(dpg: Any, presentation: Any) -> None:
    section = presentation.multistart
    if not section.available:
        dpg.set_value(
            "pyecr_diagnostics_multistart_status",
            f"MultiStart: N/A — {section.reason}",
        )
        if dpg.does_item_exist("pyecr_diagnostics_multistart_table"):
            dpg.delete_item("pyecr_diagnostics_multistart_table")
        _clear_table_copy(dpg, "pyecr_diagnostics_multistart_table", section.reason)
        return
    dpg.set_value(
        "pyecr_diagnostics_multistart_status",
        f"MultiStart spread: {section.comparable_candidates}/"
        f"{section.total_candidates} retained candidates are within 1% SSE",
    )
    _replace_diagnostics_table(
        dpg,
        table_tag="pyecr_diagnostics_multistart_table",
        parent="pyecr_diagnostics_multistart_container",
        headings=(
            "Parameter",
            "Minimum",
            "Maximum",
            "Spread (decades)",
            "Candidates",
        ),
        rows=tuple(
            (
                row.parameter,
                format_scientific(row.minimum),
                format_scientific(row.maximum),
                (
                    "N/A — nonpositive values"
                    if row.spread_decades is None
                    else format_scientific(row.spread_decades)
                ),
                str(row.candidate_count),
            )
            for row in section.rows
        ),
    )


def _format_confidence_interval(
    lower: float | None,
    upper: float | None,
) -> str:
    if lower is None or upper is None:
        return "unavailable"
    return f"[{format_scientific(lower)}, {format_scientific(upper)}]"


def _clear_table_copy(dpg: Any, table_tag: str, reason: str) -> None:
    copy_tag = f"{table_tag}_full_text"
    if dpg.does_item_exist(copy_tag):
        dpg.set_value(copy_tag, f"Unavailable: {reason}")
