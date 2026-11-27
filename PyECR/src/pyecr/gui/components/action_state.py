"""Build-once action availability and disabled-reason tooltips."""

from __future__ import annotations

from typing import Any

from pyecr.gui.theme import TOKENS


def add_action_tooltip(dpg: Any, action_tag: str) -> str:
    """Attach one persistent reason tooltip to an action widget."""

    text_tag = f"{action_tag}_disabled_tooltip"
    with dpg.tooltip(parent=action_tag):
        dpg.add_text("Available", tag=text_tag, wrap=TOKENS.wraps.narrow)
    return text_tag


def set_action_availability(
    dpg: Any, action_tag: str, *, enabled: bool, reason: str = ""
) -> None:
    """Update an existing action without rebuilding it or its tooltip."""

    dpg.configure_item(action_tag, enabled=enabled)
    tooltip_tag = f"{action_tag}_disabled_tooltip"
    if hasattr(dpg, "does_item_exist") and dpg.does_item_exist(tooltip_tag):
        dpg.set_value(
            tooltip_tag,
            "Available"
            if enabled
            else (reason or "This action is not currently available."),
        )
