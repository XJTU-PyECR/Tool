"""Reusable plot tabs widgets and display helpers."""

from __future__ import annotations
from pyecr.gui.theme import TOKENS
from pyecr.gui.components.widgets import table_style_kwargs
from pyecr.gui.components.scientific_plot import signal_axis_label, time_axis_label

from pyecr.gui.components.dependencies import Any, EcrDataset, _DATA_PREVIEW_TABLE_TAG, _DATA_PREVIEW_CONTAINER_TAG
def _replace_data_preview_table(dpg: Any, preview: Any) -> object:
    """Atomically replace the sole dynamic Data Preview table.

    The table owns one stable alias. Its columns, rows, and cells intentionally use
    DearPyGui-generated UUIDs because no callback or service needs to address them.
    """

    if dpg.does_item_exist(_DATA_PREVIEW_TABLE_TAG):
        dpg.delete_item(_DATA_PREVIEW_TABLE_TAG)
    does_alias_exist = getattr(dpg, "does_alias_exist", None)
    remove_alias = getattr(dpg, "remove_alias", None)
    if (
        callable(does_alias_exist)
        and callable(remove_alias)
        and does_alias_exist(_DATA_PREVIEW_TABLE_TAG)
    ):
        remove_alias(_DATA_PREVIEW_TABLE_TAG)

    table_id = dpg.add_table(
        parent=_DATA_PREVIEW_CONTAINER_TAG,
        tag=_DATA_PREVIEW_TABLE_TAG,
        **table_style_kwargs(),
        scrollY=True,
        scrollX=True,
        inner_width=max(400, (len(preview.columns)+1)*160),
        policy=getattr(dpg, "mvTable_SizingFixedFit", 0),
        height=TOKENS.heights.preview,
    )
    try:
        dpg.add_table_column(label="Row", parent=table_id, init_width_or_weight=65, width_fixed=True)
        for column in preview.columns:
            dpg.add_table_column(label=column, parent=table_id, init_width_or_weight=160, width_fixed=True)
        for row_number, values in zip(
            preview.row_numbers,
            preview.rows,
            strict=True,
        ):
            row_id = dpg.add_table_row(parent=table_id)
            dpg.add_text(str(row_number), parent=row_id)
            for value in values:
                dpg.add_text(
                    "" if value is None else str(value),
                    parent=row_id,
                )
    except Exception:
        if dpg.does_item_exist(_DATA_PREVIEW_TABLE_TAG):
            dpg.delete_item(_DATA_PREVIEW_TABLE_TAG)
        raise
    return table_id


def _update_processed_plot(dpg: Any, dataset: EcrDataset) -> None:
    """Push the current processed arrays to the Data-page series and refit axes."""

    dpg.set_value(
        "pyecr_data_processed_series",
        [dataset.processed_time.tolist(), dataset.processed_signal.tolist()],
    )
    dpg.configure_item(
        "pyecr_data_processed_x_axis",
        label=time_axis_label(dataset.metadata),
    )
    dpg.configure_item(
        "pyecr_data_processed_y_axis",
        label=signal_axis_label(dataset.metadata),
    )
    axis_tags = (
        "pyecr_data_processed_x_axis",
        "pyecr_data_processed_y_axis",
    )
    set_axis_limits_auto = getattr(dpg, "set_axis_limits_auto", None)
    if callable(set_axis_limits_auto):
        for axis_tag in axis_tags:
            set_axis_limits_auto(axis_tag)
    for axis_tag in axis_tags:
        dpg.fit_axis_data(axis_tag)
    operations = tuple(record.operation for record in dataset.history)
    signal_type = str(dataset.metadata.get("signal_type", "unknown")).lower()
    if dataset.metadata.get("normalization_mode") == "signed_terminal":
        interpretation = (
            "Signed-DCT normalization: terminal-referenced Y (unit 1), preserving sign; "
            "this profile is separate from standard 0-to-1 progress. Undo/Reset to the source "
            "and explicitly select an appropriate progress representation before standard Fit."
        )
    elif signal_type == "normalized" and "resistance_to_conductance" in operations:
        interpretation = (
            "Conductance decreases, normalized relaxation increases from 0 to 1."
        )
    elif signal_type == "conductance":
        interpretation = (
            "Conductance plot uses G = 1/R; increasing resistance appears as "
            "decreasing conductance."
        )
    else:
        interpretation = "Processed plot uses dataset.processed_signal."
    simulation_kind = dataset.metadata.get('source_kind')
    if simulation_kind in {'simulation_clean','simulation_observed'}:
        interpretation = simulation_kind+' · simulated source, not measured experiment. '+interpretation
    exists = getattr(dpg,'does_item_exist',lambda tag:False)
    if exists('pyecr_data_processed_plot'):
        dpg.configure_item('pyecr_data_processed_plot',label=('Processed '+simulation_kind) if simulation_kind in {'simulation_clean','simulation_observed'} else 'Processed signal')
    dpg.set_value("pyecr_data_processed_interpretation", interpretation)
