"""Responsive layout calculations based solely on centralized visual tokens."""

from __future__ import annotations

from dataclasses import dataclass

from pyecr.gui.theme import TOKENS


@dataclass(frozen=True, slots=True)
class AnalysisColumns:
    left: int
    center: int
    right: int
    results_visible: bool


@dataclass(frozen=True, slots=True)
class ErrorMapGeometry:
    plot_width: int
    plot_height: int
    left_spacer: int
    colorbar_width: int


@dataclass(frozen=True, slots=True)
class DualErrorMapGeometry:
    horizontal: bool
    plot_width: int
    plot_height: int
    colorbar_width: int


@dataclass(frozen=True, slots=True)
class AnalysisWorkspaceColumns:
    left: int
    center: int
    right: int
    results_visible: bool
    results_bottom: bool
    max_left: int


def analysis_workspace_columns(width, *, preferred_left=None, results_visible=False):
    """UIR0 client-width policy; legacy analysis_columns API stays unchanged."""
    spec=TOKENS.analysis
    width=int(width)
    default=spec.left_small if width<1200 else spec.left_compact if width<1600 else spec.left_medium if width<2200 else spec.left_large
    usable=max(spec.left_min+spec.plot_emergency_min+spec.gap,width-spec.margin)
    minimum=spec.plot_min if usable>=spec.left_min+spec.gap+spec.plot_min else spec.plot_emergency_min
    maximum=max(spec.left_min,min(spec.left_max,usable-spec.gap-minimum))
    left=max(spec.left_min,min(maximum,default if preferred_left is None else int(preferred_left)))
    right=spec.summary_width if width<2200 else spec.summary_large
    bottom=width<1600 or usable-left-right-2*spec.gap<spec.plot_min
    center=usable-left-spec.gap-(right+spec.gap if results_visible and not bottom else 0)
    return AnalysisWorkspaceColumns(left,center,right,bool(results_visible),bottom,maximum)


def disclosure_columns(width, current=None, *, parameter=False):
    """Container width, with frozen 16px hysteresis. No scientific inputs."""
    spec=TOKENS.analysis
    threshold=spec.parameter_breakpoint if parameter else spec.form_breakpoint
    if current is None:return width>=threshold
    return width>=threshold-spec.hysteresis if current else width>=threshold+spec.hysteresis


def analysis_columns(
    viewport_width: int, *, results_visible: bool | None = None
) -> AnalysisColumns:
    """Give the plot the remaining width; compact view starts with results collapsed."""

    spec = TOKENS.viewport
    if results_visible is None:
        results_visible = viewport_width >= spec.compact_breakpoint
    if viewport_width < spec.compact_breakpoint:
        left, right = spec.compact_left, spec.compact_right
    elif viewport_width < spec.large_breakpoint:
        left, right = spec.medium_left, spec.medium_right
    else:
        left, right = spec.large_left, spec.large_right
    if viewport_width < 1200:
        left, right = 300, 280
    usable = max(0, int(viewport_width) - spec.outer_margin)
    gaps = spec.column_gap * (2 if results_visible else 1)
    center = max(spec.center_min, usable - left - (right if results_visible else 0) - gaps)
    return AnalysisColumns(left, center, right, results_visible)


def analysis_plot_height(viewport_height: int) -> int:
    """Fill the visualization tab without stacking independent plots."""

    spec = TOKENS.viewport
    return max(spec.plot_height_min, int(viewport_height * spec.plot_height_fraction))


def residual_panel_height(available_height: int) -> int:
    """Reserve readable axes even when the existing child must scroll."""
    return max(TOKENS.heights.fit_residual_min,
               min(TOKENS.heights.residual_max, int(available_height * .30)))


def error_map_geometry(center_width: int, plot_height: int) -> ErrorMapGeometry:
    """Keep the map near square while reserving a fixed-width adjacent scale."""

    style = TOKENS.plot
    colorbar = TOKENS.widths.colorbar
    available = max(
        140,
        center_width - colorbar - style.map_colorbar_gap - 2 * style.map_horizontal_gutter,
    )
    plot_width = min(available, int(plot_height * style.map_width_to_height))
    left_spacer = max(
        style.map_horizontal_gutter,
        (center_width - plot_width - colorbar - style.map_colorbar_gap) // 2,
    )
    return ErrorMapGeometry(plot_width, plot_height, left_spacer, colorbar)


def dual_error_map_geometry(center_width: int, plot_height: int) -> DualErrorMapGeometry:
    """Fit two readable maps at 1920+; stack them in the scrollable center at 1366."""

    style = TOKENS.plot
    bar = TOKENS.widths.colorbar
    horizontal = center_width >= 1000
    available = (center_width - style.map_horizontal_gutter * 2) // (2 if horizontal else 1)
    width = max(180, available - bar - style.map_colorbar_gap - style.map_horizontal_gutter)
    height = min(plot_height, max(TOKENS.viewport.plot_height_min, int(width / style.map_width_to_height)))
    return DualErrorMapGeometry(horizontal, width, height, bar)
