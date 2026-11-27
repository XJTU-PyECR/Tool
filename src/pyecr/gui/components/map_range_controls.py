"""Explicit map range drafts, view-only Apply, and existing grid dispatch."""
from dataclasses import replace
from math import log10
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.plotting.map_ranges import log_range, within_bounds, auto_range
from pyecr.plotting.map_display import grid_points

P={'single':'pyecr_error_map','dual':'pyecr_dual_map'}

class MapRangeControlsMixin:
    def _build_map_range_controls(self,family):
        d=self._require_dpg();p=P[family]
        self._range_bindings=getattr(self,'_range_bindings',{})
        self._range_baselines=getattr(self,'_range_baselines',{})
        self._range_pending=getattr(self,'_range_pending',{})
        self._range_coordinates=getattr(self,'_range_coordinates',{})
        self._range_coordinates[family]='Physical'
        with d.table(header_row=False,width=-1,policy=d.mvTable_SizingStretchProp,pad_outerX=False):
            d.add_table_column(width_fixed=True,init_width_or_weight=100);d.add_table_column()
            with d.table_row():
                d.add_text('Range')
                d.add_combo(('Auto','Custom'),default_value='Auto',tag=p+'_range_mode',
                    width=-1,callback=self._map_range_edited,user_data=(family,None))
        with d.group(tag=p+'_range_inputs',show=False):
            if family=='dual':
                d.add_checkbox(label='Link H/O ranges',tag=p+'_range_link',
                               callback=self._map_range_edited,user_data=(family,'link'))
            with d.table(header_row=True,width=-1,policy=d.mvTable_SizingStretchProp,pad_outerX=False):
                for title in ('Parameter','Minimum','Maximum'):d.add_table_column(label=title)
                for process in self._range_processes(family):
                    for axis,symbol,unit in (('d','D','cm²/s'),('k','k','cm/s')):
                        with d.table_row():
                            d.add_text(f'{symbol}{"_"+process if family=="dual" else ""} ({unit})',tag=p+'_range_'+process.lower()+'_'+axis+'_title')
                            for endpoint,label in (('min','minimum'),('max','maximum')):
                                tag=self._range_tag(family,process,axis,endpoint)
                                with d.group():
                                    d.add_text(f'{symbol} {label} ({unit})',tag=tag+'_label',show=False)
                                    d.add_input_text(tag=tag,width=-1,callback=self._map_range_edited,user_data=(family,process))
        with d.collapsing_header(label='Range tools',tag=p+'_range_tools',default_open=False):
            with d.group(horizontal=True):
                d.add_text('Inputs')
                d.add_combo(('Physical','log10'),default_value='Physical',tag=p+'_range_coordinate',
                    callback=self._map_range_coordinate_changed,user_data=family,width=120)
            with d.table(header_row=False,width=-1,policy=d.mvTable_SizingStretchProp):
                d.add_table_column();d.add_table_column()
                with d.table_row():
                    d.add_button(label='Full bounds',width=-1,callback=self._map_range_full_bounds,user_data=family)
                    d.add_button(label='Reset auto',width=-1,tag=p+'_range_reset',callback=self._reset_map_range,user_data=family)
                with d.table_row():
                    d.add_button(label='Intersect bounds',width=-1,tag=p+'_range_intersection',callback=self._map_range_intersection,user_data=family)
        with d.collapsing_header(label='Use existing map view',tag=p+'_view_range_tools',default_open=False):
            with d.group(horizontal=True):
                d.add_button(label='Use current view',tag=p+'_use_current',callback=self._use_current_map_range,user_data=family)
                d.add_button(label='Apply view',tag=p+'_apply_view',callback=self._apply_map_range_view,user_data=family)
            d.add_text('Uses stored cells; no calculation.',color=TOKENS.steps.secondary)
        d.add_text('',tag=p+'_range_status',wrap=TOKENS.map_controls.wrap,show=False)
        d.add_text('',tag=p+'_range_warning',wrap=TOKENS.map_controls.wrap,color=TOKENS.colors.warning,show=False)

    @staticmethod
    def _range_processes(family):return ('single',) if family=='single' else ('H','O')

    def _range_tag(self,family,process,axis,endpoint):return P[family]+'_range_'+process.lower()+'_'+axis+'_'+endpoint

    def _has_range_controls(self,family):
        return self._require_dpg().does_item_exist(P[family]+'_range_coordinate')

    def _range_bounds(self,family):
        if family=='single':
            d,k=self._axis_bounds();return {'single':(d,k)}
        b=self._dual_map_bounds()
        return {p:(b[p.lower(),'d'],b[p.lower(),'k']) for p in ('H','O')}

    def _range_values_auto(self,family):
        d=self._require_dpg();bounds=self._range_bounds(family);fit=self.controller.analysis_session.fit_result
        half=float(d.get_value(P[family]+'_half_width'))
        return {p:tuple(auto_range(log10(fit.params[name]),half,b) for name,b in zip(
            ('D_cm2_s','k_cm_s') if p=='single' else (f'D_{p}_cm2_s',f'k_{p}_cm_s'),bounds[p])) for p in bounds}

    def _fill_range_values(self,family,ranges):
        d=self._require_dpg();physical=d.get_value(P[family]+'_range_coordinate')=='Physical'
        for process,pair in ranges.items():
            for axis,values in zip(('d','k'),pair):
                for endpoint,value in zip(('min','max'),values):
                    # Python's shortest round-trip form keeps full float64
                    # identity while showing 1e-7 rather than 17 trailing 9s.
                    d.set_value(self._range_tag(family,process,axis,endpoint),repr(float(10**value if physical else value)))

    def _range_signature(self,family):
        d=self._require_dpg();p=P[family]
        return (d.get_value(p+'_range_mode'),d.get_value(p+'_range_coordinate'),
            tuple(d.get_value(self._range_tag(family,s,a,e)) for s in self._range_processes(family) for a in ('d','k') for e in ('min','max')),
            d.get_value(p+'_resolution'),d.get_value(p+'_points'),d.get_value(p+'_half_width'))

    def _read_map_ranges(self,family,*,for_grid=False):
        d=self._require_dpg();p=P[family];mode=d.get_value(p+'_range_mode')
        if mode in ('Auto','AUTO'):return self._range_values_auto(family)
        bounds=self._range_bounds(family)
        if mode=='FULL BOUNDS':return bounds
        nodes=grid_points(d.get_value(p+'_resolution'),d.get_value(p+'_points')) if for_grid else 401
        ranges={}
        for process in self._range_processes(family):
            pair=[]
            for index,axis in enumerate(('d','k')):
                if mode=='MANUAL':
                    legacy=p+('_' if family=='single' else '_'+process.lower()+'_')+axis+'_range'
                    values=str(d.get_value(legacy)).split(',');coordinate='log10'
                else:
                    values=[d.get_value(self._range_tag(family,process,axis,e)) for e in ('min','max')]
                    coordinate=d.get_value(p+'_range_coordinate')
                pair.append(within_bounds(log_range(values,coordinate=coordinate,nodes=nodes),bounds[process][index],name=process+' '+axis.upper()))
            ranges[process]=tuple(pair)
        return ranges

    def _range_notice(self,family,message):
        d=self._require_dpg();d.set_value(P[family]+'_range_status',message)
        d.configure_item(P[family]+'_range_status',show=bool(message))

    def _map_range_edited(self,sender=None,app_data=None,user_data=None):
        family,process=user_data;d=self._require_dpg();p=P[family]
        d.configure_item(p+'_range_inputs',show=d.get_value(p+'_range_mode') not in ('Auto','AUTO'))
        if family=='dual' and d.get_value(p+'_range_link') and process:
            source='O' if process=='O' else 'H';target='H' if source=='O' else 'O'
            for a in ('d','k'):
                for e in ('min','max'):d.set_value(self._range_tag(family,target,a,e),d.get_value(self._range_tag(family,source,a,e)))
        self._range_notice(family,'Range changed. Recalculate to apply.')

    def _map_range_coordinate_changed(self,sender=None,app_data=None,user_data=None):
        family=user_data;d=self._require_dpg();old=self._range_coordinates[family];new=d.get_value(P[family]+'_range_coordinate')
        try:
            previous=list(self._range_signature(family));previous[1]=old
            unchanged=tuple(previous)==self._range_baselines.get(family)
            converted={}
            for s in self._range_processes(family):
                converted[s]=tuple(log_range([d.get_value(self._range_tag(family,s,a,e)) for e in ('min','max')],coordinate=old) for a in ('d','k'))
            self._fill_range_values(family,converted);self._range_coordinates[family]=new
            if unchanged:self._range_baselines[family]=self._range_signature(family)
            for s in self._range_processes(family):
                for a,t,u in (('d','X','cm²/s'),('k','Y','cm/s')):
                    symbol='D' if a=='d' else 'k'
                    d.set_value(P[family]+'_range_'+s.lower()+'_'+a+'_title',
                        f'{"log10 " if new=="log10" else ""}{symbol}{"_"+s if family=="dual" else ""} ({u})')
                    for e,l in (('min','minimum'),('max','maximum')):
                        d.set_value(self._range_tag(family,s,a,e)+'_label',f'{t} {l} ({"log10 " if new=="log10" else ""}{"D" if a=="d" else "k"}, {u})')
            # A unit representation change alone is not an unapplied range edit.
            self._range_notice(family,'Input coordinates converted; scientific values are unchanged.')
        except (ValueError,OverflowError) as exc:
            d.set_value(P[family]+'_range_coordinate',old);self._range_notice(family,f'Correct the numeric fields before converting: {exc}')

    def _map_range_full_bounds(self,sender=None,app_data=None,user_data=None):
        family=user_data
        try:
            ranges=self._range_bounds(family);self._fill_range_values(family,ranges)
            self._require_dpg().set_value(P[family]+'_range_mode','Custom')
            self._map_range_edited(user_data=(family,None))
        except (ValueError,RuntimeError,TypeError) as exc:self._range_notice(family,str(exc))

    def _map_range_intersection(self,sender=None,app_data=None,user_data=None):
        """Explicit user choice only. Never a hidden correction of a request."""
        family=user_data;d=self._require_dpg();p=P[family]
        try:
            bounds=self._range_bounds(family);ranges={}
            for process in self._range_processes(family):
                pair=[]
                for i,a in enumerate(('d','k')):
                    v=log_range([d.get_value(self._range_tag(family,process,a,e)) for e in ('min','max')],coordinate=d.get_value(p+'_range_coordinate'))
                    pair.append(log_range((max(v[0],bounds[process][i][0]),min(v[1],bounds[process][i][1])),coordinate='log10'))
                ranges[process]=tuple(pair)
            if family=='dual' and d.get_value(p+'_range_link'):
                common=tuple(log_range((max(ranges[s][i][0] for s in ranges),min(ranges[s][i][1] for s in ranges)),coordinate='log10') for i in (0,1))
                ranges={s:common for s in ranges}
            self._fill_range_values(family,ranges);d.set_value(p+'_range_mode','Custom')
            self._map_range_edited(user_data=(family,None))
            self._range_notice(family,'Legal intersection selected in the draft. Apply view or Recalculate map explicitly.')
        except (ValueError,RuntimeError,TypeError) as exc:self._range_notice(family,f'No usable legal intersection: {exc}')

    def _use_current_map_range(self,sender=None,app_data=None,user_data=None):
        family=user_data;d=self._require_dpg()
        try:
            values={}
            for s in self._range_processes(family):
                prefix=P[family]+('' if s=='single' else '_'+s.lower())
                values[s]=tuple(log_range(d.get_axis_limits(prefix+'_'+a+'_axis'),coordinate='log10') for a in ('x','y'))
            self._fill_range_values(family,values);d.set_value(P[family]+'_range_mode','Custom')
            self._map_range_edited(user_data=(family,None))
        except (ValueError,RuntimeError) as exc:self._range_notice(family,str(exc))

    def _reset_map_range(self,sender=None,app_data=None,user_data=None):
        family=user_data;d=self._require_dpg();p=P[family]
        d.set_value(p+'_range_mode','Auto');d.set_value(p+'_half_width',2.)
        d.configure_item(p+'_range_inputs',show=False)
        try:self._fill_range_values(family,self._range_values_auto(family))
        except (ValueError,RuntimeError,TypeError):pass
        current=self.controller.current_error_map_result if family=='single' else self.controller.current_dual_error_map_result
        if current is not None:self._apply_map_range_view(user_data=family)
        else:self._range_notice(family,'Auto range selected. Recalculate map when the parent Fit is current.')

    def _apply_map_range_view(self,sender=None,app_data=None,user_data=None):
        family=user_data
        try:
            current=self.controller.current_error_map_result if family=='single' else self.controller.current_dual_error_map_result
            if current is None:raise ValueError('A current computed grid is required. Recalculate map after a current Fit.')
            ranges=self._read_map_ranges(family)
            mode='Auto' if self._require_dpg().get_value(P[family]+'_range_mode') in ('Auto','AUTO') else 'Custom'
            options=replace(self.controller.map_view_options(family),display_ranges=tuple((p,*r) for p,r in ranges.items()),range_mode=mode)
            self.controller.set_map_view_options(family,options)
            for process,pair in ranges.items():self._apply_map_axis_range(family,process,pair)
            self._range_baselines[family]=self._range_signature(family)
            self._range_notice(family,'View applied to stored cells. Outside computed coverage: no data.')
        except (ValueError,RuntimeError,TypeError,OverflowError) as exc:self._range_notice(family,f'View rejected: {exc}')

    def _recalculate_map_ranges(self,family):
        try:
            from pyecr.diagnostics.error_map import ErrorMapConfig
            from pyecr.diagnostics.dual_error_map import DualErrorMapConfig,DualMapRangeMode
            d=self._require_dpg();p=P[family];ranges=self._read_map_ranges(family,for_grid=True)
            n=grid_points(d.get_value(p+'_resolution'),d.get_value(p+'_points'))
            half=float(d.get_value(p+'_half_width'));auto=d.get_value(p+'_range_mode') in ('Auto','AUTO')
            if family=='single':
                pair=(None,None) if auto else ranges['single']
                settings=ErrorMapConfig(n_points=n,log10_D_range=pair[0],log10_k_range=pair[1],auto_half_width_decades=half)
                task=self.controller.run_error_map(settings)
            else:
                kw={} if auto else {f'{s.lower()}_log10_{a}_range':r[i] for s,r in ranges.items() for i,a in enumerate(('D','k'))}
                settings=DualErrorMapConfig(n_points=n,range_mode=DualMapRangeMode.AUTO if auto else DualMapRangeMode.MANUAL,auto_half_width_decades=half,**kw)
                task=self.controller.run_dual_error_map(settings)
            self._range_pending[family]=(task,self._range_signature(family),settings,ranges)
            self._remember_map_attempt(family)
            self._range_notice(family,'Recalculating the frozen range. Previous complete grid retained.')
        except (ValueError,RuntimeError,TypeError,OverflowError,NotImplementedError) as exc:self._range_notice(family,f'Recalculate rejected: {exc}')

    def _refresh_range_form(self,family):
        if not self._has_range_controls(family):return
        d=self._require_dpg();c=self.controller;s=c.analysis_session;p=P[family]
        binding=(id(s),id(s.dataset),id(s.current_analysis_config),id(s.fit_result),s.fit_stale)
        if self._range_bindings.get(family)!=binding:
            self._range_bindings[family]=binding
            stored=c.map_view_options(family).display_ranges
            try:
                ranges={process:(dr,kr) for process,dr,kr in stored} or self._range_values_auto(family)
                self._fill_range_values(family,ranges);d.set_value(p+'_range_mode',c.map_view_options(family).range_mode if stored else 'Auto')
            except (ValueError,RuntimeError,TypeError):d.set_value(p+'_range_mode','Auto')
            d.configure_item(p+'_range_inputs',show=d.get_value(p+'_range_mode')=='Custom');self._range_baselines[family]=self._range_signature(family)
            self._range_pending.pop(family,None)
        pending=self._range_pending.get(family);task=c.state.tasks[GuiPage.DIAGNOSTICS]
        if pending and task.task_id==pending[0] and task.status not in ('running','pending'):
            config=s.error_map_config if family=='single' else s.dual_error_map_config
            current=c.current_error_map_result if family=='single' else c.current_dual_error_map_result
            if task.status in ('completed','warning') and config==pending[2] and current is not None:
                self._range_baselines[family]=pending[1]
                mode='Auto' if pending[1][0] in ('Auto','AUTO') else 'Custom'
                c.set_map_view_options(family,replace(c.map_view_options(family),display_ranges=tuple((p,*r) for p,r in pending[3].items()),range_mode=mode))
                for process,pair in pending[3].items():self._apply_map_axis_range(family,process,pair)
            self._range_pending.pop(family,None)
        if self._range_signature(family)!=self._range_baselines.get(family):
            text=str(d.get_value(p+'_range_status'))
            if not text.startswith(('View rejected:','Recalculate rejected:','Correct the numeric')):
                self._range_notice(family,'Range changed. Recalculate to apply.')
        current=c.current_error_map_result if family=='single' else c.current_dual_error_map_result
        for action in ('apply_view','use_current'):d.configure_item(p+'_'+action,enabled=current is not None)
        outside=[]
        if current is not None:
            grids=(current,) if family=='single' else (current.h_map,current.o_map)
            for process,grid in zip(self._range_processes(family),grids):
                prefix=p+('' if process=='single' else '_'+process.lower())
                if not d.get_item_state(prefix+'_plot').get('visible'):continue
                x,y=d.get_axis_limits(prefix+'_x_axis'),d.get_axis_limits(prefix+'_y_axis')
                if x[0]<x[1] and y[0]<y[1] and not (x[0]<=grid.fitted_point.log10_D<=x[1] and y[0]<=grid.fitted_point.log10_k<=y[1]):outside.append(process)
        d.set_value(p+'_range_warning','Fit point outside view: '+', '.join(outside)+'. The Fit is unchanged.' if outside else '')
        d.configure_item(p+'_range_warning',show=bool(outside))
        if getattr(self,'_map_focus_family',None)==family:
            note='Stored grid only; colors are objective values, not confidence intervals.'
            if outside:note+=' Fit point outside view: '+', '.join(outside)+'.'
            if current is None:note+=' Historical grid: parent Fit is out of date or unavailable.'
            d.set_value('pyecr_map_focus_note',note)
        releases=getattr(self,'_map_axis_releases',{})
        for tag,frame in tuple(releases.items()):
            if d.get_frame_count()>=frame:d.set_axis_limits_auto(tag);del releases[tag]
        self._flush_map_axis_ranges()

    def _apply_map_axis_range(self,family,process,pair):
        d=self._require_dpg();prefix=P[family]+('' if process=='single' else '_'+process.lower())
        self._map_desired_ranges=getattr(self,'_map_desired_ranges',{})
        self._map_desired_ranges[prefix]=pair
        manager=getattr(self,'_plot_views',None)
        if manager is not None:
            key=prefix+'_plot';manager.change(key,x_range=pair[0],y_range=pair[1]);manager.range_applied.add(key)
        self._flush_map_axis_ranges()

    def _flush_map_axis_ranges(self):
        d=self._require_dpg();self._map_axis_releases=getattr(self,'_map_axis_releases',{})
        desired=getattr(self,'_map_desired_ranges',{})
        for prefix,pair in tuple(desired.items()):
            if not d.get_item_state(prefix+'_plot').get('visible'):continue
            for axis,values in zip(('x','y'),pair):
                d.set_axis_limits(prefix+'_'+axis+'_axis',*values)
                self._map_axis_releases[prefix+'_'+axis+'_axis']=d.get_frame_count()+2
                from pyecr.gui.views.error_map import _decade_ticks
                d.set_axis_ticks(prefix+'_'+axis+'_axis',_decade_ticks(values))
            del desired[prefix]

    def _render_map_axis_range(self,family,process,result,options,edges):
        """Range changes never rebuild the grid/raster, and style changes never reset zoom."""
        from pyecr.plotting.map_display import padded_limits
        if not self._has_range_controls(family):
            d=self._require_dpg();prefix=P[family]+('' if process=='single' else '_'+process.lower())
            for a,e in zip(('x','y'),edges):
                if hasattr(d,'set_axis_limits'):d.set_axis_limits(prefix+'_'+a+'_axis',*padded_limits(e,options))
                else:d.fit_axis_data(prefix+'_'+a+'_axis')
            return
        pair=options.ranges(process)
        key=(family,process);seen=getattr(self,'_map_axes_seen',{})
        signature=(id(result),pair,options.padding)
        if seen.get(key)!=signature:
            pair=pair or tuple(padded_limits(e,options) for e in edges)
            # T07 native pan/zoom saved with the project takes priority on reopen.
            manager=getattr(self,'_plot_views',None)
            if manager is not None:
                manager._store();view=manager.views.get(P[family]+('' if process=='single' else '_'+process.lower())+'_plot')
                if view is not None:pair=(view.x_range or pair[0],view.y_range or pair[1])
            self._apply_map_axis_range(family,process,pair);seen[key]=signature;self._map_axes_seen=seen
