"""Final figure-card layout for frozen legends; never changes data or limits."""
from .view_state import export_legend_position,wrap_label
from .legend_labels import compact_legend_label


def place_export_legend(axis,legend,state,*,policy='follow'):
    fig=axis.figure
    if policy == 'auto':
        from dataclasses import replace
        from .legend_avoidance import avoid_curves
        # Resolve final text wrapping before testing the complete legend box.
        place_export_legend(axis, legend, replace(state, legend_mode='preset', legend_position='right'))
        fig.canvas.draw()
        chosen, evidence = avoid_curves(axis, legend.get_window_extent(fig.canvas.get_renderer()), state)
        place_export_legend(axis, legend, chosen)
        axis._pyecr_legend_placement = evidence
        return
    if policy != 'follow':
        raise ValueError('unsupported export legend policy')
    for text in legend.get_texts():
        original=getattr(text,'_pyecr_source_label',text.get_text())
        text._pyecr_source_label=compact_legend_label(original)
        text.set_text(text._pyecr_source_label)
    legend.set_frame_on(state.legend_frame)
    frame=legend.get_frame();frame.set_edgecolor('#7b838c');frame.set_linewidth(.8)
    frame.set_facecolor(axis.get_facecolor());frame.set_alpha(.97)
    legend.borderaxespad=0
    if state.legend_mode!='outside':
        legend.set_in_layout(False)
        for _ in range(3):
            fig.canvas.draw();r=fig.canvas.get_renderer();box=axis.get_window_extent(r)
            for t in legend.get_texts():
                original=getattr(t,'_pyecr_source_label',t.get_text());t._pyecr_source_label=original
                t.set_text(wrap_label(original,max(40.,box.width*.8-fig.dpi*8/72),
                    lambda s:r.get_text_width_height_descent(s,t.get_fontproperties(),False)[0]))
            fig.canvas.draw();r=fig.canvas.get_renderer();box=axis.get_window_extent(r);extent=legend.get_window_extent(r)
            if extent.width<=box.width and extent.height<=box.height:break
            w,h=fig.get_size_inches();inches=(w*max(1.,extent.width/box.width*1.05),h*max(1.,extent.height/box.height*1.05))
            if max(inches)>40:raise ValueError('Inside legend exceeds export layout budget; choose shorter display labels or a larger layout')
            fig.set_size_inches(*inches,forward=False)
        fig.canvas.draw();r=fig.canvas.get_renderer();box=axis.get_window_extent(r);extent=legend.get_window_extent(r)
        if extent.width>box.width or extent.height>box.height:raise ValueError('Inside legend has no usable plot area')
        x,y=export_legend_position((0.,0.,box.width,box.height),(extent.width,extent.height),state,fig.dpi)
        legend.set_loc('upper left');legend.set_bbox_to_anchor((x/box.width,1-y/box.height),transform=axis.transAxes)
        return
    # Freeze publication/tight/constrained layout before allocating a real strip
    # inside this axes card. Colorbar axes move with their mappable, not beneath
    # the legend. Other panels retain their own positions and ranges.
    legend.set_in_layout(False);fig.canvas.draw()
    if not hasattr(fig,'_pyecr_legend_bases'):
        fig._pyecr_legend_bases={a:a.get_position().frozen() for a in fig.axes}
        fig.set_layout_engine(None)
    bases=fig._pyecr_legend_bases
    group=[axis]+[a for a in fig.axes if getattr(getattr(a,'_colorbar',None),'mappable',None) is not None
                 and a._colorbar.mappable.axes is axis]
    x0=min(bases[a].x0 for a in group);x1=max(bases[a].x1 for a in group)
    y0=min(bases[a].y0 for a in group);y1=max(bases[a].y1 for a in group)
    horizontal=state.outside_side in ('left','right')
    for _ in range(3):
        fig.canvas.draw();r=fig.canvas.get_renderer();W,H=fig.bbox.width,fig.bbox.height
        budget=max(70.,(x1-x0)*W*(.42 if horizontal else .85))
        for t in legend.get_texts():
            original=getattr(t,'_pyecr_source_label',t.get_text());t._pyecr_source_label=original
            t.set_text(wrap_label(original,budget,lambda s:r.get_text_width_height_descent(s,t.get_fontproperties(),False)[0]))
        fig.canvas.draw();r=fig.canvas.get_renderer();ext=legend.get_window_extent(r)
        # Save backends can round hinted CJK glyphs differently from the Agg
        # measurement canvas. Keep physical outer breathing space in addition
        # to the inter-item gap, including after a save DPI change.
        guard=fig.dpi*8/72+(.04*ext.width if horizontal else .04*ext.height)
        needW=max(ext.width+12+guard+100 if horizontal else ext.width+2*guard,(x1-x0)*W)
        needH=max(ext.height+12+guard+80 if not horizontal else ext.height+2*guard,(y1-y0)*H)
        newW=max(W,needW/(x1-x0));newH=max(H,needH/(y1-y0))
        if newW<=W+.1 and newH<=H+.1:break
        inches=(newW/fig.dpi,newH/fig.dpi)
        if max(inches)>40:raise ValueError('Outside legend exceeds export layout budget; choose shorter display labels or a larger layout')
        fig.set_size_inches(*inches,forward=False)
    W,H=fig.bbox.width,fig.bbox.height;gap=12.;r=fig.canvas.get_renderer();ext=legend.get_window_extent(r)
    # Original decoration extents exclude the legend. Reserving its extra size
    # inside the original tight box preserves titles, ticks, units and panels.
    from matplotlib.transforms import Bbox
    tight=Bbox.union([a.get_tightbbox(r) for a in group]);bounds=Bbox.from_extents(x0*W,y0*H,x1*W,y1*H)
    dl,dr=max(0.,bounds.x0-tight.x0),max(0.,tight.x1-bounds.x1)
    db,dt=max(0.,bounds.y0-tight.y0),max(0.,tight.y1-bounds.y1)
    guard=fig.dpi*8/72+(.04*ext.width if horizontal else .04*ext.height)
    reserve=(ext.width+gap+guard)/W if horizontal else (ext.height+gap+guard)/H
    nx0,nx1,ny0,ny1=x0,x1,y0,y1
    if state.outside_side=='left':nx0+=reserve
    elif state.outside_side=='right':nx1-=reserve
    elif state.outside_side=='top':ny1-=reserve
    else:ny0+=reserve
    if nx1<=nx0 or ny1<=ny0:raise ValueError('Outside legend has no usable plot area')
    for a in group:
        b=bases[a];a.set_position((nx0+(b.x0-x0)/(x1-x0)*(nx1-nx0),
            ny0+(b.y0-y0)/(y1-y0)*(ny1-ny0),b.width/(x1-x0)*(nx1-nx0),b.height/(y1-y0)*(ny1-ny0)))
    align=state.outside_align
    if horizontal:
        left=nx0*W-dl-gap-ext.width if state.outside_side=='left' else nx1*W+dr+gap
        top=ny1*H-max(0.,(ny1-ny0)*H-ext.height)*align
    else:
        left=nx0*W+max(0.,(nx1-nx0)*W-ext.width)*align
        top=ny1*H+dt+gap+ext.height if state.outside_side=='top' else ny0*H-db-gap
    legend.set_loc('upper left');legend.set_bbox_to_anchor((left/W,top/H),transform=fig.transFigure)
    fig.canvas.draw()
    final=legend.get_window_extent(fig.canvas.get_renderer())
    if final.x0 < -.5 or final.y0 < -.5 or final.x1>W+.5 or final.y1>H+.5:
        raise ValueError('Outside legend would be cropped; enlarge the export layout')
