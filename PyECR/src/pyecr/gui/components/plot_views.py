"""Input-owning overlay legends on real ImPlot inner rectangles.

No scientific callbacks: polling updates only project presentation metadata.
The small custom series is a no-fit coordinate probe, never an observation.
"""
from dataclasses import replace
from math import isfinite

from pyecr.plotting.view_state import (METADATA_KEY, default_plot_view, decode_views,
                                      encode_views, freeze_views, placed_legend, POSITIONS, SIDES, wrap_label)
from pyecr.gui.theme import TOKENS, PLOT_FONT_TAG
from pyecr.plotting.legend_labels import compact_legend_label


class PlotViews:
    def __init__(self, dpg, metadata):
        self.dpg, self.metadata = dpg, metadata
        self.plots = {}
        self.rects = {}
        self.available = {}
        self.drag = None
        self.bound_metadata = None
        self.range_applied = set()
        self.last_ranges = {}
        self.range_ready = {}
        self.release_axes = []
        self.bindings = []
        self.palette_themes = {}
        self.wheel_frame = -10
        self.cards={};self.legend_rects={};self.menu_key=None;self.layout_ranges={}
        self._make_themes()
        with dpg.handler_registry() as self.input_registry:
            dpg.add_mouse_wheel_handler(callback=lambda *args: setattr(self, 'wheel_frame', dpg.get_frame_count()))
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right,callback=self._right_clicked)
        self.discover()

    def _make_themes(self):
        d=self.dpg;self.legend_themes={}
        for frame in (False,True):
            with d.theme() as theme:
                with d.theme_component(d.mvChildWindow):
                    d.add_theme_color(d.mvThemeCol_ChildBg,(*TOKENS.colors.child,250))
                    d.add_theme_color(d.mvThemeCol_Border,(135,145,156,255))
                    d.add_theme_style(d.mvStyleVar_ChildBorderSize,1 if frame else 0)
                    d.add_theme_style(d.mvStyleVar_WindowPadding,6,4)
                    d.add_theme_style(d.mvStyleVar_ItemSpacing,4,2)
                    d.add_theme_style(d.mvStyleVar_FramePadding,3,1)
            self.legend_themes[frame]=theme
        with d.theme() as self.card_theme:
            with d.theme_component(d.mvChildWindow):
                d.add_theme_style(d.mvStyleVar_WindowPadding,0,0)
                d.add_theme_style(d.mvStyleVar_ChildBorderSize,0)

    def discover(self):
        d = self.dpg
        for item in tuple(d.get_all_items()):
            if not d.get_item_info(item)['type'].endswith('mvPlot'):
                continue
            key = d.get_item_alias(item)
            if key in self.plots:
                continue
            for child in sum(d.get_item_children(item).values(), []):
                if d.get_item_info(child)['type'].endswith('mvPlotLegend'):
                    d.configure_item(child, location=d.mvPlot_Location_East)
            if not key:
                continue
            children = d.get_item_children(item, 1)
            axes = [c for c in children if d.get_item_info(c)['type'].endswith('mvPlotAxis')]
            if len(axes) < 2:
                continue
            x, y = axes[:2]
            self._series(y)
            for child in sum(d.get_item_children(item).values(), []):
                if d.get_item_info(child)['type'].endswith('mvPlotLegend'):
                    d.configure_item(child, show=False)
            probe = d.add_custom_series([0., 1.], [0., 1.], 2, parent=y,
                tag=key+'__view_probe', no_fit=True, tooltip=False,
                callback=self._coordinates, user_data=key)
            window = key+'__legend'
            with d.child_window(tag=window, parent=self._owner_window(plot=item),
                          no_scrollbar=True, no_scroll_with_mouse=True,
                          width=170,height=90,border=True,show=False):
                d.add_button(label='Legend', tag=window+'_handle',width=-1)
                with d.tooltip(window+'_handle',delay=TOKENS.plot.legend_help_delay,hide_on_activity=True):
                    d.add_text('Drag freely. Right-click the legend for position, frame and reset.')
                d.add_child_window(tag=window+'_rows',width=-1,height=-1,border=False)
            if d.does_item_exist(PLOT_FONT_TAG):
                d.bind_item_font(window,PLOT_FONT_TAG)
            self.plots[key] = (item, x, y, probe, window)
            self._make_menu(key)

    def _make_menu(self,key):
        d=self.dpg;tag=key+'__legend_menu'
        with d.window(tag=tag,popup=True,show=False,autosize=True,no_move=True,no_saved_settings=True):
            with d.menu(label='Position'):
                for position in POSITIONS:
                    d.add_menu_item(label=position.replace('_',' ').capitalize(),
                        tag=tag+'_'+position,check=True,callback=lambda s,a,data:self.select_position(*data),user_data=(key,position))
            with d.menu(label='Outside'):
                for side in SIDES:
                    d.add_menu_item(label=side.capitalize(),tag=tag+'_outside_'+side,check=True,
                        callback=lambda s,a,data:self.select_outside(*data),user_data=(key,side))
            d.add_menu_item(label='Free drag',tag=tag+'_free',check=True,
                callback=lambda s,a,k:self.select_free(k),user_data=key)
            d.add_separator()
            d.add_menu_item(label='Frame',tag=tag+'_frame',check=True,default_value=True,
                callback=lambda s,a,k:self._menu_change(k,legend_frame=bool(a)),user_data=key)
            d.add_menu_item(label='Show legend',tag=tag+'_visible',check=True,default_value=True,
                callback=lambda s,a,k:self._menu_change(k,legend_visible=bool(a)),user_data=key)
            d.add_menu_item(label='Reset legend',callback=lambda s,a,k:self._menu_reset(k),user_data=key)

    def _close_menu(self,key):
        self.dpg.configure_item(key+'__legend_menu',show=False);self.menu_key=None

    def _menu_change(self,key,**kwargs):self.change(key,**kwargs);self._close_menu(key)
    def _menu_reset(self,key):self.reset(key);self._close_menu(key)
    def select_position(self,key,position):self._menu_change(key,legend_mode='preset',legend_position=position,legend_visible=True)
    def select_outside(self,key,side):self._menu_change(key,legend_mode='outside',outside_side=side,legend_visible=True)
    def select_free(self,key):self._menu_change(key,legend_mode='free',legend_visible=True)

    def open_menu(self,key):
        self._store();s=self.views.get(key,default_plot_view());d=self.dpg;tag=key+'__legend_menu'
        for p in POSITIONS:d.set_value(tag+'_'+p,s.legend_mode=='preset' and s.legend_position==p)
        for side in SIDES:d.set_value(tag+'_outside_'+side,s.legend_mode=='outside' and s.outside_side==side)
        for suffix,value in (('free',s.legend_mode=='free'),('frame',s.legend_frame),('visible',s.legend_visible)):
            d.set_value(tag+'_'+suffix,value)
        d.set_item_pos(tag,d.get_mouse_pos(local=False))
        d.configure_item(tag,show=True);self.menu_key=key

    def _right_clicked(self,*args):
        mouse=self.dpg.get_mouse_pos(local=False)
        for key,(x,y,w,h) in reversed(tuple(self.legend_rects.items())):
            window=self.plots[key][4]
            hovered=(self.dpg.get_item_state(window).get('hovered') or self.dpg.get_item_state(window+'_rows').get('hovered'))
            if self.dpg.get_item_configuration(window)['show'] and hovered and x<=mouse[0]<=x+w and y<=mouse[1]<=y+h:
                self.open_menu(key);return

    def _owner_window(self, plot):
        d=self.dpg
        p=plot
        while p:
            if d.get_item_info(p)['type'].endswith('mvWindowAppItem'):
                return p
            p=d.get_item_info(p).get('parent')
        raise ValueError('plot requires an owning window')

    def forget(self,key):
        """Release a dynamic result plot; persisted display state stays intact."""
        binding=self.plots.get(key)
        if binding is None:return
        plot,x,y,probe,window=binding
        self._unreserve(key)
        self.plots.pop(key)
        if self.dpg.does_item_exist(y):
            for item in self.dpg.get_item_children(y,1):self.available.pop(item,None)
        self.available={i:v for i,v in self.available.items() if self.dpg.does_item_exist(i)}
        if self.dpg.does_item_exist(window):self.dpg.delete_item(window)
        for tag in (key+'__legend_menu',probe):
            if self.dpg.does_item_exist(tag):self.dpg.delete_item(tag)
        self.legend_rects.pop(key,None)
        self.layout_ranges.pop(key,None)
        if self.menu_key==key:self.menu_key=None
        if hasattr(self,'_rows_'+key):delattr(self,'_rows_'+key)
        for cache in (self.rects,self.last_ranges,self.range_ready):cache.pop(key,None)
        self.range_applied.discard(key);self.release_axes=[a for a in self.release_axes if a not in (x,y)]
        if self.drag is not None and self.drag[0]==key:self._finish_drag()

    def show_all(self, show=True):
        for key in self.plots:
            self.change(key, legend_visible=show)

    def reset_all(self):
        for key in self.plots:
            self.reset(key)

    def _color(self, item, index):
        d = self.dpg
        theme = d.get_item_info(item).get('theme')
        if theme:
            for comp in d.get_item_children(theme, 1):
                for child in d.get_item_children(comp, 1):
                    if d.get_item_info(child)['type'].endswith('mvThemeColor'):
                        cfg = d.get_item_configuration(child)
                        if cfg.get('category') == d.mvThemeCat_Plots and cfg.get('target') in (d.mvPlotCol_Line, d.mvPlotCol_MarkerFill):
                            return tuple(d.get_value(child))
        return tuple(v*255 for v in d.get_colormap_color(d.mvPlotColormap_Deep, index % 10))

    def _bind_palette_color(self, item, info, index):
        """Give unstyled curves and their overlay one explicit color source."""
        if not info['type'].endswith(('mvLineSeries', 'mvScatterSeries')) or info.get('theme'):
            return
        d = self.dpg
        palette = index % 10
        if palette not in self.palette_themes:
            color = [v*255 for v in d.get_colormap_color(d.mvPlotColormap_Deep, palette)]
            with d.theme() as theme:
                for kind in (d.mvLineSeries, d.mvScatterSeries):
                    with d.theme_component(kind):
                        for target in (d.mvPlotCol_Line, d.mvPlotCol_MarkerFill, d.mvPlotCol_MarkerOutline):
                            d.add_theme_color(target, color, category=d.mvThemeCat_Plots)
            self.palette_themes[palette] = theme
        d.bind_item_theme(item, self.palette_themes[palette])

    def _coordinates(self, sender, data, key):
        if not data or len(data) < 3 or len(data[1]) < 2:
            return
        a, b = data[1][:2], data[2][:2]
        rect = (min(a), min(b), abs(a[1]-a[0]), abs(b[1]-b[0]))
        if all(isfinite(v) for v in rect) and rect[2] > 1 and rect[3] > 1:
            self.rects[key] = rect

    def _store(self):
        metadata = self.metadata()
        wire = metadata.get(METADATA_KEY)
        if metadata is not self.bound_metadata or wire is not getattr(self, '_stored_views', None):
            self.bound_metadata = metadata
            self._stored_views = wire
            self.views = decode_views(wire)
            self.range_applied.clear()
            self.last_ranges.clear()
            self.range_ready.clear()
            self.layout_ranges.clear()
            self.drag = None
        return metadata

    def change(self, key, **kwargs):
        metadata = self._store()
        if 'legend_anchor' in kwargs and 'legend_mode' not in kwargs:
            kwargs['legend_mode'] = 'free'
        self.views[key] = replace(self.views.get(key, default_plot_view()), **kwargs)
        metadata[METADATA_KEY] = encode_views(self.views)
        self._stored_views = metadata[METADATA_KEY]

    def freeze(self):
        """Capture effective current placements, including unedited chart defaults."""
        self._store()
        self._finish_drag()
        states = {key: default_plot_view() for key in self.plots}
        states.update(self.views)
        return freeze_views(encode_views(states))

    def _finish_drag(self):
        """Commit one completed gesture, instead of encoding every pointer move."""
        if self.drag is None:return
        self.bound_metadata[METADATA_KEY]=encode_views(self.views)
        self._stored_views = self.bound_metadata[METADATA_KEY]
        self.drag=None

    def reset(self, key):
        self._store()
        state = self.views.get(key, default_plot_view()).reset_legend()
        self.change(key,**{k:getattr(state,k) for k in ('legend_anchor','legend_visible','legend_mode',
            'legend_position','outside_side','outside_align','legend_frame')})

    def toggle(self, key, series_id):
        self._store()
        hidden = set(self.views.get(key, default_plot_view()).hidden_series)
        hidden.symmetric_difference_update([series_id])
        layers = set(self.views.get(key, default_plot_view()).hidden_layers)
        if series_id.endswith(('cross_h', 'grid_cross_h')):
            vertical = series_id[:-1]+'v'
            if series_id in hidden:
                layers.add(vertical)
            else:
                layers.discard(vertical)
        self.change(key, hidden_series=tuple(hidden), hidden_layers=tuple(layers))

    def before_refresh(self):
        # Restore application-requested visibility before renderers decide which
        # result/mode is applicable; reapply user hiding only after rendering.
        d = self.dpg
        # Return application-facing plot dimensions before its normal layout
        # pass. Reserve strips once afterwards, never progressively shrink.
        for key,(_,cfg) in tuple(self.cards.items()):
            if key in self.plots and d.does_item_exist(self.plots[key][0]):d.configure_item(self.plots[key][0],**cfg)
        for item, show in self.available.items():
            if d.does_item_exist(item):
                d.configure_item(item, show=show)

    def _unreserve(self,key):
        if key not in self.cards:return
        d=self.dpg;card,cfg=self.cards.pop(key);plot=self.plots[key][0]
        if d.does_item_exist(card):
            if d.does_item_exist(plot):
                self._remember_layout_ranges(key)
                d.move_item(plot,parent=d.get_item_parent(card),before=card)
                d.configure_item(plot,**cfg);d.reset_pos(plot)
            d.delete_item(card)

    def _remember_layout_ranges(self,key):
        d=self.dpg;_,x,y,_,_=self.plots[key]
        state=self.views.get(key)
        if (not d.get_item_state(self.plots[key][0]).get('visible') and state is not None
                and state.x_range is not None and state.y_range is not None):
            # A map below its controls may never have rendered before Expand.
            # Preserve the pending captured range, not ImPlot's [0,1] placeholder.
            self.layout_ranges[key]=((state.x_range,state.y_range),3)
            return
        if key not in self.range_applied:
            # Initial [0,1] is ImPlot's placeholder, not a user-selected view.
            # A persisted Outside can reparent before the first real axis fit.
            self.layout_ranges[key]=(None,3)
            return
        limits=(tuple(d.get_axis_limits(x)),tuple(d.get_axis_limits(y)))
        if all(isfinite(v) for p in limits for v in p) and all(a<b for a,b in limits):
            self.layout_ranges[key]=(limits,3)
            for axis,pair in zip((x,y),limits):d.set_axis_limits(axis,*pair)

    def _reserve(self,key,state,size):
        """A real child card owns the extra strip; adjacent plots/colorbars stay out."""
        d=self.dpg;plot=self.plots[key][0]
        if state.legend_mode!='outside':self._unreserve(key);return None
        cfg=d.get_item_configuration(plot);wanted={k:cfg[k] for k in ('width','height')}
        if key not in self.cards:
            card=key+'__view_card';parent=d.get_item_parent(plot)
            d.add_child_window(tag=card,parent=parent,before=plot,**wanted,border=False,horizontal_scrollbar=True,
                no_scroll_with_mouse=True,always_use_window_padding=True)
            d.bind_item_theme(card,self.card_theme);self._remember_layout_ranges(key);d.move_item(plot,parent=card)
        else:card=self.cards[key][0]
        self.cards[key]=(card,wanted);d.configure_item(card,**wanted,show=cfg.get('show',True))
        measured=d.get_item_rect_size(card);previous=d.get_item_rect_size(plot)
        w=measured[0] if measured[0]>1 else max(280,previous[0]);h=measured[1] if measured[1]>1 else max(180,previous[1])
        lw,lh=size;gap=10
        if state.outside_side in ('left','right'):
            pw=max(140,w-lw-gap);ph=max(120,h)
            px=lw+gap if state.outside_side=='left' else 0;py=0
            zone=(0,0,lw,h) if state.outside_side=='left' else (pw+gap,0,lw,h)
        else:
            pw=max(140,w);ph=max(120,h-lh-gap)
            px=0;py=lh+gap if state.outside_side=='top' else 0
            zone=(0,0,w,lh) if state.outside_side=='top' else (0,ph+gap,w,lh)
        d.configure_item(plot,width=int(pw),height=int(ph));d.set_item_pos(plot,(px,py))
        origin=self._window_origin(card);scroll=d.get_item_state(card).get('scroll_pos',(0,0))
        return (origin[0]-scroll[0]+zone[0],origin[1]-scroll[1]+zone[1],zone[2],zone[3])

    def _series(self, y):
        d = self.dpg
        rows = []
        palette_index = 0
        for index, item in enumerate(d.get_item_children(y, 1)):
            info, config = d.get_item_info(item), d.get_item_configuration(item)
            kind = info['type'].split('::')[-1]
            if kind in ('mvCustomSeries', 'mvHeatSeries', 'mvImageSeries') or not kind.endswith('Series'):
                continue  # heatmap uses its own color bar, not a categorical legend
            # Sensitivities are created/recreated after discovery. Bind before
            # drawing, rather than relying on ImPlot's cached automatic colors.
            # Probes do not consume a color; hidden/empty curves keep theirs.
            self._bind_palette_color(item, info, palette_index)
            palette_index += 1
            show = bool(config.get('show', True))
            self.available[item] = show
            label = config.get('label', '').split('##')[0]
            if not show or not label:
                continue
            values = d.get_value(item)
            if not show or not label or not values or not any(len(v) for v in values if hasattr(v, '__len__')):
                continue
            key = d.get_item_alias(item) or f'{kind}:{index}:{label}'
            rows.append((item, key, label))
        return rows

    def _visible_rect(self, plot, rect, *, require_plot=True):
        """Overlay cannot escape scrolling children, collapsed panels or viewport."""
        d = self.dpg
        if not d.get_item_configuration(plot).get('show',True):return None
        if require_plot and not d.get_item_state(plot).get('visible'):
            return None
        left, top, width, height = rect
        right, bottom = left+width, top+height
        p = d.get_item_info(plot).get('parent')
        while p:
            state = d.get_item_state(p)
            info = d.get_item_info(p)
            if not d.get_item_configuration(p).get('show',True):return None
            if info['type'].endswith(('mvCollapsingHeader','mvTreeNode')) and not d.get_value(p):return None
            if info['type'].endswith('mvTab'):
                selected=d.get_value(info['parent'])
                alias=d.get_item_alias(p)
                identities=(p,alias,d.get_alias_id(alias)) if alias else (p,)
                if selected not in identities:return None
            if info['type'].endswith(('mvChildWindow', 'mvWindowAppItem')) and state.get('rect_size'):
                a = self._window_origin(p)
                b = state.get('rect_max', (a[0]+state['rect_size'][0], a[1]+state['rect_size'][1]))
                left, top, right, bottom = max(left,a[0]), max(top,a[1]), min(right,b[0]), min(bottom,b[1])
            p = info.get('parent')
        if right<=left or bottom<=top:
            return None
        return (left,top,right-left,bottom-top)

    def _window_origin(self, item):
        d = self.dpg
        pos = list(d.get_item_state(item).get('pos', (0, 0)))
        p = d.get_item_info(item).get('parent')
        while p:
            info = d.get_item_info(p)
            if info['type'].endswith(('mvChildWindow', 'mvWindowAppItem')):
                state = d.get_item_state(p)
                offset, scroll = state.get('pos', (0, 0)), state.get('scroll_pos', (0, 0))
                pos = [pos[i]+offset[i]-scroll[i] for i in range(2)]
            p = info.get('parent')
        return pos

    def refresh(self):
        d = self.dpg
        self._store()
        for axis in self.release_axes:
            d.set_axis_limits_auto(axis)
        self.release_axes.clear()
        menu_active=any(d.get_item_configuration(k+'__legend_menu')['show'] for k in self.plots)
        for key, (plot, x, y, probe, window) in tuple(self.plots.items()):
            if not d.does_item_exist(plot):self.forget(key);continue
            owner=self._owner_window(plot)
            if d.get_item_parent(window)!=owner:
                self._remember_layout_ranges(key)
                d.move_item(window,parent=owner)
            if key in self.cards:d.configure_item(self.cards[key][0],show=d.get_item_configuration(plot).get('show',True))
            if key in self.layout_ranges:
                pairs,remaining=self.layout_ranges[key]
                for index,axis in enumerate((x,y)):
                    if pairs is None:
                        d.set_axis_limits_auto(axis)
                        if remaining:d.fit_axis_data(axis)
                    elif remaining:d.set_axis_limits(axis,*pairs[index])
                    else:d.set_axis_limits_auto(axis)
                if remaining:self.layout_ranges[key]=(pairs,remaining-1)
                else:del self.layout_ranges[key]
            limits = (tuple(d.get_axis_limits(x)), tuple(d.get_axis_limits(y)))
            if all(isfinite(v) for pair in limits for v in pair) and all(a < b for a,b in limits):
                d.set_value(probe, [list(limits[0]), list(limits[1])])
            state = self.views.get(key, default_plot_view())
            if state.legend_mode!='outside' or not state.legend_visible:self._unreserve(key)
            previous, count = self.range_ready.get(key, (None, 0))
            count = count+1 if previous == limits else 0
            self.range_ready[key] = (limits,count)
            visible_for_restore=d.get_item_state(plot).get('visible')
            if not visible_for_restore and key in self.cards and key in self.rects:
                visible_for_restore=self._visible_rect(plot,self.rects[key],require_plot=False) is not None
            if key not in self.range_applied and key not in self.layout_ranges and count>=2 and visible_for_restore:
                for axis, value in ((x, state.x_range), (y, state.y_range)):
                    if value is not None:
                        d.set_axis_limits(axis, *value)
                        self.release_axes.append(axis)
                self.range_applied.add(key)
            rows = self._series(y)
            rect = self.rects.get(key)
            clip = self._visible_rect(plot, rect,require_plot=key not in self.cards) if rect else None
            font=PLOT_FONT_TAG if d.does_item_exist(PLOT_FONT_TAG) else 0
            text_size=lambda label:d.get_text_size(label,font=font)
            measure=text_size('M')
            if not rect or measure is None or not clip:
                if self.drag is not None and self.drag[0]==key:self._finish_drag()
                d.configure_item(window,show=False);self.legend_rects.pop(key,None)
                continue
            card=self.cards.get(key)
            outer=d.get_item_rect_size(card[0] if card else plot)
            width=outer[0] if state.legend_mode=='outside' else clip[2]
            height=outer[1] if state.legend_mode=='outside' else clip[3]
            cap=max(110,width*(.45 if state.legend_mode=='outside' and state.outside_side in ('left','right') else .9))
            compact=[compact_legend_label(label) for _,_,label in rows]
            preferred=max([text_size('Legend')[0]+24]+[text_size(label)[0]+measure[1]+34 for label in compact])
            lw=min(cap,preferred);budget=max(24,lw-measure[1]-34)
            labels=[wrap_label(label,budget,lambda s:text_size(s)[0]) for label in compact]
            content=sum(text_size(label)[1]+4 for label in labels)
            lh=min(max(54,height*(.6 if state.legend_mode=='outside' else .9)),content+measure[1]+24)
            size=(lw,lh)
            signature = tuple((i, k, label, self._color(i,n)) for n,(i,k,label) in enumerate(rows)), state.hidden_series, round(budget),measure
            if getattr(self, '_rows_'+key, None) != signature:
                d.delete_item(window+'_rows', children_only=True)
                for n,((item, series, label),shown) in enumerate(zip(rows,labels)):
                    visible = series not in state.hidden_series
                    with d.group(horizontal=True, parent=window+'_rows'):
                        marker_size=measure[1]
                        d.add_color_button(default_value=self._color(item,n), no_tooltip=True,
                            width=marker_size, height=marker_size,
                            callback=lambda s, a, data: self.toggle(*data), user_data=(key,series))
                        button = d.add_selectable(label=shown, default_value=visible,
                            width=budget,height=text_size(shown)[1]+2,
                            callback=lambda s, a, data: self.toggle(*data), user_data=(key, series))
                        with d.tooltip(button):
                            d.add_text(label)
                setattr(self, '_rows_'+key, signature)
            # Generic toolbar wrapping must not stack this marker/label pair;
            # the legend computes its own bounded text/scroll budget above.
            for row in d.get_item_children(window+'_rows',1):
                d.configure_item(row,horizontal=True)
            for item, series, label in rows:
                if series in state.hidden_series or series in state.hidden_layers:
                    d.configure_item(item, show=False)
            for item in d.get_item_children(y, 1):
                if d.get_item_alias(item) in state.hidden_layers:
                    self.available.setdefault(item, bool(d.get_item_configuration(item).get('show', True)))
                    d.configure_item(item, show=False)
            visible = bool(rows and state.legend_visible and clip)
            d.configure_item(window, show=visible,width=int(lw),height=int(lh),border=state.legend_frame)
            d.bind_item_theme(window,self.legend_themes[state.legend_frame])
            zone=self._reserve(key,state if visible else replace(state,legend_mode='free'),size)
            if zone is not None:clip=self._visible_rect(plot,zone,require_plot=False)
            # Persist only actual user pan/zoom, never an automatic result fit.
            mouse=d.get_mouse_pos(local=False)
            over_legend=any(a<=mouse[0]<=a+w and b<=mouse[1]<=b+h for a,b,w,h in self.legend_rects.values())
            if not over_legend and self.drag is None and not menu_active and d.is_item_hovered(plot) and (d.is_mouse_button_down(d.mvMouseButton_Left) or d.is_mouse_button_down(d.mvMouseButton_Right) or d.get_frame_count()-self.wheel_frame <= 2):
                if limits != self.last_ranges.get(key) and all(a < b for a,b in limits):
                    self.change(key, x_range=limits[0], y_range=limits[1])
            self.last_ranges[key] = limits
            if not visible or clip is None:
                if self.drag is not None and self.drag[0]==key:self._finish_drag()
                d.configure_item(window,show=False)
                self.legend_rects.pop(key,None)
                continue
            size=(min(size[0],clip[2]),min(size[1],clip[3]))
            d.configure_item(window,width=max(1,int(size[0])),height=max(1,int(size[1])))
            if zone is not None:
                pos=(zone[0]+max(0,zone[2]-size[0])*state.outside_align,
                     zone[1]+max(0,zone[3]-size[1])*state.outside_align)
            else:pos=placed_legend(rect,size,state)
            def clip_position(position):
                return (min(max(position[0],clip[0]),clip[0]+clip[2]-size[0]),
                        min(max(position[1],clip[1]),clip[1]+clip[3]-size[1]))
            pos = clip_position(pos)
            handle = window+'_handle'
            released=d.get_item_state(handle).get('deactivated',False)
            active=d.is_item_active(handle)
            if active or (self.drag is not None and self.drag[0]==key):
                mouse = d.get_mouse_pos(local=False)
                if self.drag is None:
                    # ImGui retains the button-down origin in its drag delta.
                    # The first observed active frame may already contain the
                    # final move; do not adopt that late position as the origin.
                    delta=d.get_mouse_drag_delta() if d.is_mouse_button_dragging(d.mvMouseButton_Left,0.) else (0.,0.)
                    self.drag = (key, (mouse[0]-delta[0],mouse[1]-delta[1]), pos)
                if self.drag[0] == key:
                    _, origin, initial = self.drag
                    new = clip_position((initial[0]+mouse[0]-origin[0], initial[1]+mouse[1]-origin[1]))
                    anchor = ((new[0]-rect[0])/rect[2], (new[1]-rect[1])/rect[3])
                    self.views[key]=replace(self.views.get(key,default_plot_view()),legend_anchor=anchor,legend_mode='free')
                    # Rebase at the clamped pixel position each frame: reversing
                    # after an edge hit follows immediately, with no dead travel.
                    self.drag=(key,mouse,new);pos=new
                if released or (not active and not d.is_mouse_button_down(d.mvMouseButton_Left)):self._finish_drag()
            origin=self._window_origin(owner)
            scroll=d.get_item_state(owner).get('scroll_pos',(0,0))
            d.set_item_pos(window, (pos[0]-origin[0]+scroll[0],pos[1]-origin[1]+scroll[1]))
            self.legend_rects[key]=(*pos,*size)

