"""Shared Error Map view controls; no objective or fitting calls."""
import traceback
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.plotting.map_display import MapViewOptions, PALETTES, RESOLUTIONS, grid_points, palette_colors
from pyecr.gui.components.map_range_controls import MapRangeControlsMixin

PREFIX = {"single":"pyecr_error_map", "dual":"pyecr_dual_map"}

class MapControlsMixin(MapRangeControlsMixin):
    def _remember_map_attempt(self, family):
        attempts = getattr(self, "_map_attempt_ids", {})
        attempts[family] = self.controller.state.tasks[GuiPage.DIAGNOSTICS].task_id
        self._map_attempt_ids = attempts

    def _map_attempt_summary(self, family):
        task = self.controller.state.tasks[GuiPage.DIAGNOSTICS]
        remembered = getattr(self, "_map_attempt_ids", {}).get(family)
        if remembered is None or remembered != task.task_id:
            return ""
        if task.status in {"cancelled", "error", "failed"}:
            return f"Last Recalculate: {task.status}. Previous captured grid retained."
        return ""

    def _build_map_current_fit(self, family):
        dpg = self._require_dpg(); prefix = PREFIX[family]
        dpg.add_text("1 · Source", tag=prefix+"_fit_heading")
        dpg.add_text("", tag=prefix+"_fit_context", wrap=TOKENS.map_controls.wrap)

    def _refresh_map_instructions(self, family):
        """Read captured state only; never prepare/evaluate a new grid."""
        from pyecr.model_display import display_label, BY_ID
        dpg = self._require_dpg(); prefix = PREFIX[family]
        from pyecr.gui.components.compact_panels import resize_pair,content_width
        for section,parts in (('calculation',('range','grid')),('display',('metric','style')),('advanced',('tools','options'))):
            resize_pair(dpg,prefix+'_'+section+'_columns',tuple(prefix+'_'+part+'_panel' for part in parts))
        for suffix in ('fit_context','status','captured_grid','quality_banner','display_info','hover',
                       'range_status','range_warning','view_error'):
            tag=prefix+'_'+suffix
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag,wrap=content_width(dpg,tag))
        self._refresh_range_form(family)
        if not dpg.does_item_exist(prefix+"_fit_context"):
            return  # Legacy headless widget doubles.
        session = self.controller.analysis_session
        config, fit = session.current_analysis_config, session.fit_result
        required = "03 M2_SINGLE_DK" if family == "single" else "06 M5_DUAL_H2O_DK"
        model = "none" if config is None else (display_label(config.model_id) if config.model_id.value in BY_ID else config.model_id.value)
        ready = self.controller.workflow.can_error_map if family == "single" else self.controller.workflow.can_dual_error_map
        dpg.set_value(prefix+"_fit_context", f"{'Current' if ready else 'Unavailable'} fit: {model}. "
                      + ("" if ready else f"Requires {required} and a current supported fit. ")
                      + (f"N={len(fit.residuals)} full fit-eligible observations." if fit is not None else "Run Fit first."))
        if dpg.does_item_exist(prefix+"_resolution"):
            try:
                n=grid_points(dpg.get_value(prefix+"_resolution"),dpg.get_value(prefix+"_points"))
                dpg.set_value(prefix+"_budget",f"{n} × {n} per map · {n*n*(2 if family=='dual' else 1):,} cells total")
            except (TypeError,ValueError):
                dpg.set_value(prefix+"_budget","Invalid grid input. Correct it before Recalculate.")
        result = self.controller.current_error_map_result if family == "single" else self.controller.current_dual_error_map_result
        if result is None:
            captured = ""
        else:
            grids = (result,) if family == "single" else (result.h_map, result.o_map)
            captured = "Displayed: " + " | ".join(f"{len(grid.log10_D)} × {len(grid.log10_k)}" for grid in grids)
        dpg.set_value(prefix+"_captured_grid", captured)
        dpg.configure_item(prefix+"_captured_grid",show=bool(captured))

    def _build_map_view_controls(self, family):
        dpg=self._require_dpg();prefix=PREFIX[family];view=self.controller.map_view_options(family)
        if not dpg.does_item_exist("pyecr_map_focus_window"):
            with dpg.window(label="Expanded Error Map",tag="pyecr_map_focus_window",show=False,modal=True,
                            width=TOKENS.map_controls.focus_width,height=TOKENS.map_controls.focus_height,on_close=self._close_map_focus):
                dpg.add_button(label="Return to settings",callback=self._close_map_focus)
                dpg.add_button(label='Cancel map',tag='pyecr_map_focus_cancel',show=False,
                               callback=lambda:self.controller.cancel_page(GuiPage.DIAGNOSTICS))
                dpg.add_text("View of the same captured grid. Colors are objective values, not confidence intervals.",tag='pyecr_map_focus_note',wrap=TOKENS.map_controls.focus_wrap)
                dpg.add_group(tag="pyecr_map_focus_content")
            self._owned_layout_roots.add("pyecr_map_focus_window")
        if not dpg.does_item_exist("pyecr_map_colormaps"):
            with dpg.colormap_registry(tag="pyecr_map_colormaps"):
                for palette in PALETTES:
                    for reverse in (False, True):
                        name=MapViewOptions(palette=palette, reverse=reverse).cmap_name
                        colors=palette_colors(name).astype(int).tolist()
                        dpg.add_colormap(colors,qualitative=False,tag="pyecr_palette_"+name)
        if not dpg.does_item_exist("pyecr_map_grid_min_theme"):
            with dpg.theme(tag="pyecr_map_grid_min_theme"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line,(230,75,70,255),category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight,2.,category=dpg.mvThemeCat_Plots)
        from pyecr.gui.components.compact_panels import field
        with dpg.group(horizontal=True,tag=prefix+'_display_columns'):
            dpg.add_child_window(tag=prefix+'_metric_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True)
            dpg.add_child_window(tag=prefix+'_style_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True)
        with dpg.table(parent=prefix+'_metric_panel',tag=prefix+'_display_fields',header_row=False,width=-1,policy=dpg.mvTable_SizingStretchProp):
            # Explicit weights prevent ImGui's automatic column measurement
            # from shrinking Palette to its arrow beside the longer fields.
            dpg.add_table_column(init_width_or_weight=1.7,width_stretch=True)
            dpg.add_table_column(init_width_or_weight=1.,width_stretch=True)
            with dpg.table_row():
                tags=[(prefix+'_metric','Metric')]
                if family=='single':tags.append((prefix+'_clip','Contrast'))
                for tag,title in tags:
                    with dpg.group() as cell:
                        dpg.add_text(title)
                        dpg.move_item(tag,parent=cell);dpg.configure_item(tag,label='',width=-1)
                if family=='dual':
                    field(dpg,'Contrast',dpg.add_combo,items=('Full','P95','P99'),default_value=view.contrast,
                          tag=prefix+'_contrast',callback=self._map_view_changed,user_data=family)
            with dpg.table_row():
                field(dpg,'Palette',dpg.add_combo,items=PALETTES,default_value=view.palette,
                      tag=prefix+'_palette',callback=self._map_view_changed,user_data=family)
                field(dpg,'Rendering',dpg.add_combo,items=('Cells','Smooth'),
                      default_value='Smooth' if not self.analysis_map_metadata(family) else view.sampling,
                      tag=prefix+'_sampling',callback=self._map_view_changed,user_data=family)
            if family=='dual':
                with dpg.table_row():
                    with dpg.group() as cell:
                        dpg.add_text('Color scale')
                        dpg.move_item(prefix+'_color_scale',parent=cell)
                        dpg.configure_item(prefix+'_color_scale',label='',width=-1)
                    dpg.move_item(prefix+'_grid_min',parent=prefix+'_metric_panel')
        if family=='dual':dpg.delete_item(prefix+'_toolbar')
        dpg.push_container_stack(prefix+'_style_panel')
        dpg.add_checkbox(label='Reverse colors',default_value=view.reverse,tag=prefix+'_reverse',
                         callback=self._map_view_changed,user_data=family)
        with dpg.collapsing_header(label="Color limits / export quality",tag=prefix+'_color_quality',default_open=False):
            with dpg.table(header_row=False,width=-1,policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=1.,width_stretch=True)
                dpg.add_table_column(init_width_or_weight=1.5,width_stretch=True)
                with dpg.table_row():
                    field(dpg,'Color range',dpg.add_combo,items=('Auto','Manual'),default_value='Auto' if view.color_range is None else 'Manual',
                          tag=prefix+'_color_mode',callback=self._map_view_changed,user_data=family)
                    field(dpg,'Color min,max',dpg.add_input_text,default_value='' if view.color_range is None else ','.join(map(str,view.color_range)),
                          tag=prefix+'_color_range',callback=self._map_view_changed,user_data=family)
                with dpg.table_row():
                    field(dpg,'PNG DPI',dpg.add_combo,items=('150','300','600'),default_value=str(view.image_dpi),
                          tag=prefix+'_dpi',callback=self._map_view_changed,user_data=family)
                    field(dpg,'Axis margin',dpg.add_input_float,default_value=view.padding,min_value=0,max_value=.5,
                          min_clamped=True,max_clamped=True,tag=prefix+'_padding',callback=self._map_view_changed,user_data=family)
            dpg.add_text("Smooth is a 501-pixel bilinear view of stored cells; hover/minimum/CSV use evaluated cells. "
                         "Extra axis margin is blank outside computed coverage. PNG quality does not increase numerical resolution.",wrap=TOKENS.map_controls.wrap)
        dpg.pop_container_stack()
        self._map_view_seen=getattr(self,"_map_view_seen",{})
        self._map_view_seen[family]=None
        dpg.add_text("",tag=prefix+"_view_error",show=False,color=TOKENS.colors.error,wrap=TOKENS.wraps.map)

    def analysis_map_metadata(self,family):
        return self.controller.analysis_session.project_metadata.get("error_map_view",{}).get(family)

    def _map_options_from_controls(self,family):
        dpg=self._require_dpg();prefix=PREFIX[family]
        if not dpg.does_item_exist(prefix+"_palette"):return self.controller.map_view_options(family)
        manual=dpg.get_value(prefix+"_color_mode")=="Manual"
        limits=None
        if manual:
            limits=tuple(float(x.strip()) for x in str(dpg.get_value(prefix+"_color_range")).split(','))
            if len(limits)!=2:raise ValueError("color range requires min,max")
        clip_tag="pyecr_error_map_clip" if family=="single" else prefix+"_contrast"
        return MapViewOptions(metric=dpg.get_value(prefix+"_metric"),
                              shared_scale=family=="dual" and dpg.get_value(prefix+"_color_scale")=="Shared",
                              show_grid_min=bool(dpg.get_value(prefix+("_show_grid_min" if family=="single" else "_grid_min"))),
                              palette=dpg.get_value(prefix+"_palette"),reverse=bool(dpg.get_value(prefix+"_reverse")),
                              sampling=dpg.get_value(prefix+"_sampling"),contrast=dpg.get_value(clip_tag),color_range=limits,
                              image_dpi=int(dpg.get_value(prefix+"_dpi")),padding=float(dpg.get_value(prefix+"_padding")),
                              display_ranges=self.controller.map_view_options(family).display_ranges,
                              range_mode=self.controller.map_view_options(family).range_mode)

    def _map_view_changed(self,sender=None,app_data=None,user_data=None):
        family=user_data or ("dual" if str(sender).startswith("pyecr_dual") else "single")
        try:
            options=self._map_options_from_controls(family)
            self.controller.set_map_view_options(family,options)
            self._map_view_seen=getattr(self,"_map_view_seen",{})
            self._map_view_seen[family]=options.to_dict()
            error_tag=PREFIX[family]+"_view_error"
            if self._require_dpg().does_item_exist(error_tag):
                self._require_dpg().set_value(error_tag,"");self._require_dpg().configure_item(error_tag,show=False)
            if family=="single":self._render_error_map_plot(force=True)
            else:self._render_dual_error_map_plot(force=True)
        except Exception as exc:
            error_tag=PREFIX[family]+"_view_error"
            if self._require_dpg().does_item_exist(error_tag):
                self._require_dpg().set_value(error_tag,f"Display options: {exc}; previous valid view retained.")
                self._require_dpg().configure_item(error_tag,show=True)
            # A display input must not replace a running scientific task's
            # status with "error", accidentally enabling another Generate.
            self.controller.record_notice(GuiPage.DIAGNOSTICS,"warning",f"Map display options: {exc}")

    def _map_options_for_render(self,family):
        """Restore saved presentation metadata without resetting pending edits."""
        dpg=self._require_dpg();prefix=PREFIX[family]
        if not dpg.does_item_exist(prefix+"_palette"):return self.controller.map_view_options(family)
        current=self.analysis_map_metadata(family);seen=getattr(self,"_map_view_seen",{})
        if current is not None and current!=seen.get(family):
            view=self.controller.map_view_options(family)
            for suffix,value in (("palette",view.palette),("reverse",view.reverse),("sampling",view.sampling),
                                  ("color_mode","Auto" if view.color_range is None else "Manual"),
                                  ("color_range","" if view.color_range is None else ','.join(map(str,view.color_range))),
                                  ("dpi",str(view.image_dpi)),("padding",view.padding)):
                dpg.set_value(prefix+'_'+suffix,value)
            dpg.set_value("pyecr_error_map_clip" if family=="single" else prefix+"_contrast",view.contrast)
            dpg.set_value(prefix+"_metric",view.metric)
            dpg.set_value(prefix+("_show_grid_min" if family=="single" else "_grid_min"),view.show_grid_min)
            if family=="dual":dpg.set_value(prefix+"_color_scale","Shared" if view.shared_scale else "Independent")
            seen[family]=current
        try:view=self._map_options_from_controls(family)
        except (ValueError,TypeError):return self.controller.map_view_options(family)
        # First displayed map also records its view for Save/Open and Capture.
        if current!=view.to_dict():
            self.controller.set_map_view_options(family,view);seen[family]=view.to_dict()
        return view

    def _bind_map_colors(self,family,options):
        dpg=self._require_dpg();palette="pyecr_palette_"+options.cmap_name
        if not dpg.does_item_exist(palette):return
        processes=("",) if family=="single" else ("_h","_o")
        prefix=PREFIX[family]
        for process in processes:
            dpg.bind_colormap(prefix+process+"_plot",palette)
            dpg.configure_item(prefix+process+"_colorbar",colormap=palette)

    def _build_map_resolution_controls(self,family):
        dpg=self._require_dpg();prefix=PREFIX[family]
        with dpg.table(header_row=False,width=-1,policy=dpg.mvTable_SizingStretchProp,pad_outerX=False):
            dpg.add_table_column(width_fixed=True,init_width_or_weight=100);dpg.add_table_column()
            with dpg.table_row():
                dpg.add_text('Grid')
                dpg.add_combo((*RESOLUTIONS,"Custom"),default_value="Standard",width=-1,
                          tag=prefix+"_resolution",callback=self._map_grid_changed,user_data=family)
        dpg.add_input_int(label="Points / axis",default_value=201,min_value=3,max_value=401,min_clamped=False,max_clamped=False,
                             tag=prefix+"_points",width=TOKENS.map_controls.custom_width,enabled=False,show=False,callback=self._map_grid_changed,user_data=family)
        dpg.add_text(f"81 × 81 per map · {6561*(2 if family=='dual' else 1):,} cells total",
                     tag=prefix+"_budget",wrap=TOKENS.map_controls.wrap)
        with dpg.collapsing_header(label='More settings',tag=prefix+'_calculation_advanced',default_open=False):
            with dpg.group(horizontal=True,tag=prefix+'_advanced_columns'):
                with dpg.child_window(tag=prefix+'_tools_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    dpg.move_item(prefix+'_range_tools',parent=prefix+'_tools_panel')
                    dpg.move_item(prefix+'_view_range_tools',parent=prefix+'_tools_panel')
                with dpg.child_window(tag=prefix+'_options_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    with dpg.table(header_row=False,width=-1,policy=dpg.mvTable_SizingStretchProp):
                        dpg.add_table_column()
                        with dpg.table_row():
                            from pyecr.gui.components.compact_panels import field
                            field(dpg,'Auto half-width (decades)',dpg.add_input_float,default_value=2.,min_value=.1,max_value=8.,
                                  min_clamped=False,max_clamped=False,tag=prefix+'_half_width',
                                  callback=self._map_calculation_changed,user_data=family)
                    dpg.add_text("Recalculate uses full fit data; it does not refit.",
                                 tag=prefix+"_calculation_note",wrap=TOKENS.map_controls.wrap)

    def _map_calculation_changed(self,sender=None,app_data=None,user_data=None):
        family=user_data or ("dual" if str(sender).startswith("pyecr_dual") else "single")
        self._refresh_map_instructions(family)

    def _map_grid_changed(self,sender=None,app_data=None,user_data=None):
        dpg=self._require_dpg();family=user_data or "single";prefix=PREFIX[family]
        custom=dpg.get_value(prefix+"_resolution")=="Custom"
        dpg.configure_item(prefix+"_points",enabled=custom,show=custom)
        try:
            n=grid_points(dpg.get_value(prefix+"_resolution"),dpg.get_value(prefix+"_points"))
            dpg.set_value(prefix+"_budget",f"{n} × {n} per map · {n*n*(2 if family=='dual' else 1):,} cells total")
            self._refresh_map_instructions(family)
        except Exception as exc:
            self.controller.record_notice(GuiPage.DIAGNOSTICS,'warning',str(exc))
            if self._has_range_controls(family):self._range_notice(family,f'Recalculate rejected: {exc}')

    def _focus_map(self,sender=None,app_data=None,user_data=None):
        family=user_data or "single";dpg=self._require_dpg()
        visual="pyecr_error_map_visual_row" if family=="single" else "pyecr_dual_map_plots"
        self._map_focus_parent=dpg.get_item_parent(visual);self._map_focus_visual=visual;self._map_focus_family=family
        dpg.move_item(visual,parent="pyecr_map_focus_content")
        dpg.configure_item("pyecr_map_focus_window",show=True,pos=(30,30))
        self._resize_map_focus()

    def _resize_map_focus(self):
        family=getattr(self,"_map_focus_family",None)
        if family is None:return
        dpg=self._require_dpg();width=max(TOKENS.map_controls.minimum_width,dpg.get_viewport_client_width()-TOKENS.map_controls.margin);height=max(TOKENS.map_controls.minimum_height,dpg.get_viewport_client_height()-TOKENS.map_controls.margin)
        dpg.configure_item("pyecr_map_focus_window",width=width,height=height)
        plot_height=max(TOKENS.map_controls.plot_min,height-TOKENS.map_controls.chrome)
        if family=="single":
            from pyecr.gui.layout import error_map_geometry
            g=error_map_geometry(width-36,plot_height)
            dpg.configure_item("pyecr_error_map_left_spacer",width=g.left_spacer)
            dpg.configure_item("pyecr_error_map_plot",width=g.plot_width,height=plot_height)
            dpg.configure_item("pyecr_error_map_colorbar",width=g.colorbar_width,height=plot_height)
        else:
            from pyecr.gui.layout import dual_error_map_geometry
            g=dual_error_map_geometry(width-36,plot_height)
            dpg.configure_item("pyecr_dual_map_plots",horizontal=g.horizontal)
            for process in ('h','o'):
                dpg.configure_item(f"pyecr_dual_map_{process}_plot",width=g.plot_width,height=g.plot_height)
                dpg.configure_item(f"pyecr_dual_map_{process}_colorbar",width=g.colorbar_width,height=g.plot_height)

    def _close_map_focus(self,*args):
        dpg=self._require_dpg()
        if getattr(self,"_map_focus_family",None) is not None:
            dpg.move_item(self._map_focus_visual,parent=self._map_focus_parent)
        self._map_focus_family=None
        dpg.configure_item("pyecr_map_focus_window",show=False)
        self._resize_analysis_columns()
