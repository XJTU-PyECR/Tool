"""U05 measured Reports geometry; no capture, catalog, serialization or science."""
from pyecr.gui.theme import TOKENS,FONT_TAG
from math import ceil

class ReportsGeometryMixin:
    def _reports_text_size(self,text,width=-1):
        d=self._require_dpg();font=FONT_TAG if d.does_item_exist(FONT_TAG) else 0
        scale=d.get_global_font_scale()
        size=d.get_text_size(str(text),wrap_width=width/scale if width>0 else width,font=font)
        # DPG 2.3.1 get_text_size ignores GlobalFontScale while rendered
        # controls grow. Account for it once (see U05 font probe).
        if size is not None:return (size[0]*scale,size[1]*scale)
        # DPG has no font atlas metrics before its first rendered frame. This
        # build-only estimate is replaced by actual font metrics on refresh.
        line=TOKENS.typography.body_pt*scale;length=len(str(text))*line*.52
        return (min(length,width) if width>0 else length,line*max(1,ceil(length/width)) if width>0 else line)

    def _reports_list_height(self,items,width,limit):
        d=self._require_dpg();padding=TOKENS.report.list_padding;spacing=TOKENS.report.gap
        line=self._reports_text_size('Ag')[1]
        total=2*padding
        for text,checkbox in items:
            height=self._reports_text_size(text,max(1,width-2*padding-(line+spacing if checkbox else 0)))[1]
            total+=max(line,height)+(spacing if checkbox else spacing/2)
        return max(1,min(int(limit),int(total))) if items else 1

    def _refresh_reports_geometry(self):
        d=self._require_dpg();t=TOKENS.report;pad=TOKENS.steps.padding
        width=max(1,d.get_viewport_client_width()-2*t.page_inset)
        char=self._reports_text_size('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')[0]/62
        line=self._reports_text_size('Ag')[1]
        content_min=max(self._reports_text_size('Analysis   Batch   Pair   DCT   Simulation')[0]+2*pad,
                        t.content_readable_chars*char+2*pad+line)
        destination_min=t.destination_readable_chars*char+2*pad
        breakpoint=content_min+destination_min+t.column_gap
        wide=width>=breakpoint
        left=int(max(content_min,min(width*t.content_share,width-t.column_gap-destination_min))) if wide else width
        right=width-left-t.column_gap if wide else width
        d.configure_item('pyecr_reports_columns',horizontal=wide,horizontal_spacing=t.column_gap)
        for tag,value in (('pyecr_reports_setup',left),('pyecr_reports_output',right)):
            d.configure_item(tag,width=value)
        for key,value in (('reports_results',left),('reports_content',left),('reports_destination',right)):
            d.configure_item('pyecr_step_panel_'+key,width=value)
        inner=max(1,left-2*pad)
        top=d.get_item_state('pyecr_reports_anchor').get('rect_min',(0,150))[1]
        available=max(line*6,d.get_viewport_client_height()-top-t.footer)
        module=str(d.get_value('pyecr_reports_module_tabs') or 'pyecr_reports_module_Analysis').removeprefix('pyecr_reports_module_')
        source_items=[(v.label,True) for v in self._reports_choices if v.module==module]
        if not source_items:source_items=[(getattr(self,'_reports_empty_module_text',{}).get(module,''),False)]
        source_cap=min(available*t.source_height_fraction,t.source_visible_rows*(line+t.gap)+2*t.list_padding)
        source_height=self._reports_list_height(source_items,inner,source_cap)
        d.configure_item('pyecr_reports_sources',height=source_height)
        rows=getattr(self,'_reports_list_items',())
        figure_cap=min(available*t.figure_height_fraction,t.figure_visible_rows*(line+t.gap)+2*t.list_padding)
        figure_height=self._reports_list_height(rows,inner,figure_cap)
        d.configure_item('pyecr_reports_figures',show=bool(rows),height=figure_height)
        for tag in ('pyecr_reports_count','pyecr_reports_source_count','pyecr_reports_source_warning'):
            d.configure_item(tag,wrap=inner)
        for tag in ('pyecr_reports_destination_reason','pyecr_reports_status','pyecr_reports_receipt','pyecr_reports_export_summary','pyecr_report_picker_notice'):
            d.configure_item(tag,wrap=max(1,right-2*pad))
        d.configure_item('pyecr_report_picker_notice',show=bool(d.get_value('pyecr_report_picker_notice')))
        for tag in getattr(self,'_reports_wrapped_titles',()):
            d.configure_item(tag,wrap=max(1,inner-line-t.gap-2*t.list_padding))
        for tag,_ in getattr(self,'_reports_figure_titles',()):
            if d.does_item_exist(tag):d.configure_item(tag,wrap=max(1,inner-line-t.gap-2*t.list_padding))
        for tag in getattr(self,'_reports_unavailable_texts',()):
            if d.does_item_exist(tag):d.configure_item(tag,wrap=max(1,inner))
        self._reports_layout_metrics=dict(mode='columns' if wide else 'stack',available_width=width,font_scale=d.get_global_font_scale(),
            measured_character_width=char,line_height=line,breakpoint=breakpoint,content_min=content_min,destination_min=destination_min,
            left=left,right=right,content_share=left/(left+right) if wide else None,source_height=source_height,figure_height=figure_height,
            figure_rows=len(rows),available_height=available)
