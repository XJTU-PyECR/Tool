"""Responsive widgets and complete-text inspection; no analysis state edits."""
from __future__ import annotations

from time import monotonic
import os
from pyecr.gui.theme import TOKENS, FONT_TAG, PLOT_FONT_TAG, SECTION_FONT_TAG, SECONDARY_FONT_TAG, NAVIGATION_FONT_TAG
from pyecr.gui.fonts import font_for_text


def bounded_width(available: int, preferred: int, *, gutter: int = 28) -> int:
    return max(80, min(max(80, available-gutter), preferred if preferred > 0 else available-gutter))


class DisplayLayoutMixin:
    def _build_display_menu(self) -> None:
        dpg = self._require_dpg()
        from pyecr.gui import theme
        self._display_font_plan = theme.FONT_PLAN
        self._display_fields = {}
        self._display_rows = {}
        self._display_scan_at = 0.0
        self._display_glyph_gaps = {}
        self._owned_layout_roots = set()
        self._owned_form_fields = {}
        self._owned_form_rows = {}
        self._owned_form_texts = set()
        from pyecr.gui.components.compact_fields import CompactFields
        self._compact_fields=CompactFields(self)
        with dpg.menu(label="Display"):
            dpg.add_menu_item(label="View / copy full display text", callback=self._show_display_text)
            dpg.add_menu_item(label="Font diagnostics", callback=self._show_font_diagnostics)
            with dpg.menu(label='Plot legends'):
                dpg.add_menu_item(label='Show legends', callback=lambda: self._plot_views.show_all())
                dpg.add_menu_item(label='Hide legends', callback=lambda: self._plot_views.show_all(False))
                dpg.add_menu_item(label='Reset legends', callback=lambda: self._plot_views.reset_all())
        with dpg.window(label="Full display text — select and copy with Ctrl+A / Ctrl+C",
                        tag="pyecr_display_text_window", show=False, modal=True,
                        width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            dpg.add_text("Read-only display snapshot. Scientific values and arrays are unchanged.", wrap=TOKENS.wraps.project)
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("pyecr_display_text_window", show=False))
            dpg.add_input_text(tag="pyecr_display_full_text", multiline=True, readonly=True,
                               width=-1, height=-1)
        with dpg.window(label="Font selection and coverage", tag="pyecr_font_diagnostics_window",
                        show=False, width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("pyecr_font_diagnostics_window", show=False))
            body = dpg.add_text("Time Signal Conductivity Data Processing 0123456789 cm²/s S/cm Ω µ ± −")
            science = dpg.add_text("α β γ δ Δ λ μ σ τ ∂  ₀₁₂₃₄₅₆₇₈₉  ⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ ± − ×  cm²/s")
            if dpg.does_item_exist(FONT_TAG):
                dpg.bind_item_font(body, FONT_TAG)
            if dpg.does_item_exist(PLOT_FONT_TAG):
                dpg.bind_item_font(science, PLOT_FONT_TAG)
            dpg.add_input_text(tag="pyecr_font_diagnostics_text", default_value=self._display_font_plan.describe(),
                               readonly=True, multiline=True, width=-1, height=-1)

    def _show_display_text(self, *args) -> None:
        dpg = self._require_dpg()
        lines = ["Full display text (includes collapsed panels and other pages)."]
        for item in dpg.get_all_items():
            info = dpg.get_item_info(item)
            kind = info['type'].split('::')[-1]
            if kind not in ('mvText', 'mvInputText', 'mvCombo', 'mvCollapsingHeader'):
                continue
            alias = dpg.get_item_alias(item)
            if alias.startswith(('pyecr_display_', 'pyecr_font_diagnostics')):
                continue
            label = dpg.get_item_label(item)
            value = dpg.get_value(item)
            if isinstance(value, str) and value:
                lines.append(f"[{alias or label or kind}]\n{value}")
            elif label:
                lines.append(label)
        dpg.set_value('pyecr_display_full_text', '\n\n'.join(lines))
        self._size_display_window('pyecr_display_text_window')
        dpg.configure_item('pyecr_display_text_window', show=True)

    def _show_font_diagnostics(self, *args) -> None:
        text = self._display_font_plan.describe()
        if self._display_glyph_gaps:
            text += "\nUnresolved whole-widget glyphs (text preserved):\n" + "\n".join(
                f"{tag}: {codes}" for tag,codes in self._display_glyph_gaps.values())
        self._require_dpg().set_value('pyecr_font_diagnostics_text', text)
        self._size_display_window('pyecr_font_diagnostics_window')
        self._require_dpg().configure_item('pyecr_font_diagnostics_window', show=True)

    def _display_font_warning(self):
        plan = self._display_font_plan
        if plan.body is None:
            return 'CJK font unavailable: glyphs are not verified. Display > Font diagnostics; configure PYECR_UI_FONT.'
        if os.environ.get(TOKENS.typography.font_env, '').strip() and plan.body.origin != 'PYECR_UI_FONT':
            return 'Configured UI font unavailable or missing glyphs; using checked fallback. Display > Font diagnostics.'
        return ''

    def _size_display_window(self, tag) -> None:
        dpg = self._require_dpg()
        width, height = dpg.get_viewport_width(), dpg.get_viewport_height()
        dpg.configure_item(tag, width=max(400, min(1000, width-80)), height=max(300, height-120), pos=(30, 50))

    def _refresh_display_layout(self, *, force=False) -> None:
        """Coalesced structural scan; existing callbacks/tags/values are retained.

        Fields get labels above them. Horizontal groups stack when their real
        preferred widths cannot fit. No scientific arrays are read or written.
        """
        dpg = self._require_dpg()
        if not hasattr(dpg, 'get_all_items') or not hasattr(self, '_display_fields'):
            return
        size = (dpg.get_viewport_width(), dpg.get_viewport_height())
        model_children = tuple(dpg.get_item_children('pyecr_model_schema_container', 1) or [])
        state = self.controller.state
        content = (model_children, id(self.controller.analysis_session.dataset),
                   id(self.controller.analysis_session.current_analysis_config),id(state.fit_presentation),
                   id(state.diagnostics_presentation), id(self.controller.session.import_preview))
        now = monotonic()
        unchanged = (size == getattr(self, '_display_size', None)
                and content == getattr(self, '_display_content', None))
        if (not force and unchanged
                and now-self._display_scan_at < .25):
            return
        # Poll the cheap item inventory for dynamic tables/plots, rather than
        # rewalking every ancestor and rebinding fonts four times per second.
        items = tuple(dpg.get_all_items())
        if not force and unchanged and items == getattr(self, '_display_items_snapshot', None):
            self._display_scan_at = now
            return
        self._display_size, self._display_scan_at, self._display_content = size, now, content
        self._display_layout_epoch = getattr(self,'_display_layout_epoch',0)+1
        self._compact_fields._top_cache = {}
        self._compact_fields._excluded_cache = {}
        alive = set(items)
        self._display_fields = {k:v for k,v in self._display_fields.items() if k in alive}
        self._display_rows = {k:v for k,v in self._display_rows.items() if k in alive}
        self._display_glyph_gaps = {}

        def available(item):
            parent = dpg.get_item_parent(item)
            while parent:
                cfg = dpg.get_item_configuration(parent)
                kind = dpg.get_item_info(parent)['type'].split('::')[-1]
                width = cfg.get('width', 0)
                if kind in ('mvChildWindow', 'mvWindow', 'mvTable') and width > 0:
                    return min(int(width), int(size[0])-TOKENS.viewport.outer_margin)
                parent = dpg.get_item_parent(parent)
            return int(size[0])-TOKENS.viewport.outer_margin

        def page_owned(item):
            while item:
                if dpg.get_item_alias(item) in self._owned_layout_roots:return True
                item=dpg.get_item_parent(item)
            return False

        field_types = ('mvInputText', 'mvInputFloat', 'mvInputDouble', 'mvInputInt', 'mvCombo', 'mvSliderFloat', 'mvSliderDouble', 'mvSliderInt')
        for item in items:
            info = dpg.get_item_info(item)
            kind = info['type'].split('::')[-1]
            cfg = dpg.get_item_configuration(item)
            if kind in field_types:
                if page_owned(item):continue
                if item not in self._display_fields:
                    label = cfg.get('label', '')
                    parent = dpg.get_item_parent(item)
                    if not label or not parent:
                        continue
                    wrapper = dpg.add_group(parent=parent, before=item)
                    heading = dpg.add_text(label, parent=wrapper, wrap=TOKENS.wraps.rail)
                    dpg.move_item(item, parent=wrapper)
                    dpg.configure_item(item, label='')
                    self._display_fields[item] = (wrapper, heading, cfg.get('width', 0), label)
                wrapper, heading, preferred, label = self._display_fields[item]
                # Keep conditional/manual fields and their labels in sync.
                dpg.configure_item(wrapper, show=cfg.get('show', True))
                width = bounded_width(available(item), preferred)
                dpg.configure_item(item, width=width)
                dpg.configure_item(heading, wrap=width)
            elif kind == 'mvText':
                # Table cells have their own fixed widths and horizontal scroll.
                parent_type = dpg.get_item_info(dpg.get_item_parent(item))['type'] if dpg.get_item_parent(item) else ''
                if 'mvTableRow' not in parent_type and not page_owned(item):
                    dpg.configure_item(item, wrap=bounded_width(available(item), 0))
                text = dpg.get_value(item)
                if isinstance(text, str) and text:
                    font = font_for_text(self._display_font_plan, text)
                    explicit=info.get('font')
                    roles=(SECTION_FONT_TAG,SECONDARY_FONT_TAG,NAVIGATION_FONT_TAG)
                    reserved=item in self._compact_fields.labels or self._compact_fields.top(item) or explicit in roles or any(dpg.does_item_exist(t) and explicit==dpg.get_alias_id(t) for t in roles)
                    if font is not None and not reserved:
                        dpg.bind_item_font(item, font)
                    elif font is None:
                        faces = [f for f in (self._display_font_plan.body, self._display_font_plan.scientific) if f is not None]
                        missing = min((f.missing(text) for f in faces), key=len, default=text)
                        codes = ' '.join(f'U+{ord(c):04X}' for c in dict.fromkeys(missing) if not c.isspace())
                        self._display_glyph_gaps[item] = (dpg.get_item_alias(item) or str(item), codes)
            elif kind == 'mvGroup' and cfg.get('horizontal'):
                if not page_owned(item):self._display_rows.setdefault(item, True)
            elif kind in ('mvCollapsingHeader','mvButton') and not self._compact_fields.top(item):
                if kind=='mvCollapsingHeader' and dpg.get_item_alias(item).startswith('pyecr_stage_'):continue
                if '__legend' not in dpg.get_item_alias(item) and dpg.does_item_exist(SECONDARY_FONT_TAG):dpg.bind_item_font(item,SECONDARY_FONT_TAG)
            elif kind in ('mvPlot', 'mvColorScale'):
                # Plot title, axes and legends share one face. Inspect labels
                # only, never series arrays; CJK titles/units can use the UI
                # face when the scientific face lacks their glyphs.
                labels = [cfg.get('label', '')]
                children = list(dpg.get_item_children(item, 1) or [])
                while children:
                    child = children.pop()
                    labels.append(dpg.get_item_label(child))
                    children.extend(dpg.get_item_children(child, 1) or [])
                font = font_for_text(self._display_font_plan, ''.join(labels), scientific=True)
                if font is not None:
                    dpg.bind_item_font(item, font)

        fields_by_wrapper = {v[0]:v for v in self._display_fields.values()}
        for row in self._display_rows:
            if page_owned(row):continue
            children = dpg.get_item_children(row, 1) or []
            # Map plot layout has its own exact geometry, not form wrapping.
            if any(any(kind in dpg.get_item_info(child)['type'] for kind in
                       ('mvPlot', 'mvColorScale', 'mvChildWindow')) for child in children):
                continue
            total = 0
            for child in children:
                cfg = dpg.get_item_configuration(child)
                if not cfg.get('show', True):
                    continue
                field = fields_by_wrapper.get(child)
                width = bounded_width(available(row), field[2]) if field else cfg.get('width', 0)
                if width <= 0:
                    rect = dpg.get_item_rect_size(child)
                    width = rect[0] or (len(cfg.get('label', ''))*8+16)
                total += width + TOKENS.spacing.item_spacing[0]
            dpg.configure_item(row, horizontal=total <= available(row)-28)
        self._compact_fields.discover(items)
        self._display_items_snapshot = tuple(dpg.get_all_items())
