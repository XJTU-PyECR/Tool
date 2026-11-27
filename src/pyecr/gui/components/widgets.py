"""Small reusable widget styles; no widget construction at refresh time."""

from __future__ import annotations

from pyecr.gui.theme import TOKENS


def table_style_kwargs() -> dict[str, bool]:
    """Shared border/header policy for preview and scientific result tables."""

    style = TOKENS.table
    return {
        "header_row": style.header_row,
        "borders_innerH": style.borders_inner_h,
        "borders_innerV": style.borders_inner_v,
        "borders_outerH": style.borders_outer_h,
        "borders_outerV": style.borders_outer_v,
        "resizable": style.resizable,
    }
