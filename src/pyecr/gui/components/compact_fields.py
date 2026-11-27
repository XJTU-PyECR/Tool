"""Measured label/value rows. No parsing into requests or callback invocation."""
from dataclasses import dataclass
from math import isfinite
from pyecr.gui.theme import TOKENS, SECONDARY_FONT_TAG

FIELDS=('mvInputText','mvInputFloat','mvInputDouble','mvInputInt','mvCombo',
        'mvSliderFloat','mvSliderDouble','mvSliderInt')


def row_geometry(available,label_width,control_min,*,short=True):
    """Prefer an aligned inline row only when both full label and value fit."""
    spec=TOKENS.form;gap=TOKENS.spacing.item_spacing[0]
    column=max(spec.label_column,label_width)
    inline=short and column+gap+control_min<=available
    return inline,column if inline else available,max(1,available-column-gap) if inline else max(1,available)


@dataclass(frozen=True)
class FieldRow:
    wrapper: int
    heading: int
    scalar: bool
    spacer: int


class CompactFields:
    def __init__(self,app):
        self.app=app;self.d=app._require_dpg();self.rows={};self.labels=set();self.metrics={}
        self._top_cache={};self._excluded_cache={};self._layout_cache={}

    def top(self,item):
        d=self.d;trail=[]
        while item:
            if item in self._top_cache:
                result=self._top_cache[item];break
            trail.append(item)
            if d.get_item_alias(item) in ('pyecr_information_controls','pyecr_main_menu','pyecr_header_actions'):
                result=True;break
            item=d.get_item_parent(item)
        else:result=False
        self._top_cache.update((node,result) for node in trail)
        return result

    def _excluded(self,item):
        d=self.d;trail=[]
        while item:
            if item in self._excluded_cache:
                result=self._excluded_cache[item];break
            trail.append(item)
            info=d.get_item_info(item);kind=info['type'].split('::')[-1];alias=d.get_item_alias(item)
            if kind in ('mvTable','mvPlot','mvFileDialog','mvMenuBar','mvMenu') or '__legend' in alias or self.top(item):
                result=True;break
            item=info.get('parent')
        else:result=False
        self._excluded_cache.update((node,result) for node in trail)
        return result

    def adopt(self,item,wrapper,heading):
        d=self.d
        if item in self.rows:return
        cfg=d.get_item_configuration(item);kind=d.get_item_info(item)['type'].split('::')[-1]
        scalar=not cfg.get('readonly') and not cfg.get('multiline')
        if kind=='mvInputText':
            # Numeric text stays verbatim; this only classifies its layout.
            # Names, paths, notes and explicit arrays retain the full line.
            try:scalar=scalar and isfinite(float(str(d.get_value(item)).strip()))
            except (ValueError,TypeError):scalar=False
        spacer=d.add_spacer(parent=wrapper,before=item,width=TOKENS.spacing.item_spacing[0],show=False)
        self.rows[item]=FieldRow(wrapper,heading,scalar,spacer);self.labels.add(heading)
        if d.does_item_exist(SECONDARY_FONT_TAG):d.bind_item_font(heading,SECONDARY_FONT_TAG)

    def discover(self,items):
        """Adopt existing owned wrappers, then explicit text/field sibling pairs."""
        d=self.d
        for item,(wrapper,heading,*_) in self.app._owned_form_fields.items():self.adopt(item,wrapper,heading)
        for item,(wrapper,heading,*_) in self.app._display_fields.items():
            if not self._excluded(item):self.adopt(item,wrapper,heading)
        for item in items:
            if item in self.rows or not d.does_item_exist(item):continue
            info=d.get_item_info(item);kind=info['type'].split('::')[-1]
            if kind not in FIELDS or self._excluded(item):continue
            cfg=d.get_item_configuration(item)
            if cfg.get('multiline') or cfg.get('readonly'):continue
            parent=info.get('parent');siblings=list(d.get_item_children(parent,1) or [])
            index=siblings.index(item)
            if not index:continue
            heading=siblings[index-1]
            if not d.get_item_info(heading)['type'].endswith('mvText'):continue
            label=d.get_value(heading)
            if not isinstance(label,str) or not label.strip() or '\n' in label or len(label)>TOKENS.form.label_max_chars:continue
            # Existing horizontal action rows already have their own layout.
            if d.get_item_configuration(parent).get('horizontal') and len(siblings)>2:continue
            # Never adopt an application-owned visibility group as our wrapper
            # (e.g. Simulation's conditional first-positive-time group).
            wrapper=d.add_group(parent=parent,before=heading)
            d.move_item(heading,parent=wrapper);d.move_item(item,parent=wrapper)
            self.adopt(item,wrapper,heading)

    def _available(self,item,row):
        d=self.d
        def info(node):
            if node not in self._info_cache:self._info_cache[node]=d.get_item_info(node)
            return self._info_cache[node]
        def config(node):
            if node not in self._config_cache:self._config_cache[node]=d.get_item_configuration(node)
            return self._config_cache[node]
        owned=item in self.app._owned_form_fields
        def constraints(parent):
            if not parent:return d.get_viewport_client_width()-TOKENS.viewport.outer_margin,1
            key=(parent,owned)
            if key in self._constraint_cache:return self._constraint_cache[key]
            node_info=info(parent);kind=node_info['type'].split('::')[-1]
            bound,columns=constraints(node_info.get('parent'))
            if kind in ('mvChildWindow','mvWindowAppItem','mvCollapsingHeader'):
                if parent not in self._state_cache:self._state_cache[parent]=d.get_item_state(parent)
                state=self._state_cache[parent];width=(state.get('rect_size') or (0,0))[0]
                if width>0:bound=min(bound,width-TOKENS.form.content_gutter)
                elif kind!='mvCollapsingHeader':
                    width=config(parent).get('width',0)
                    if width>0:bound=min(bound,width-TOKENS.form.content_gutter)
            elif kind=='mvGroup':
                cfg=config(parent);width=cfg.get('width',0)
                if width>0:bound=min(bound,width)
                children=d.get_item_children(parent,1) or []
                if not owned and cfg.get('horizontal') and len(children)>1 and all(info(v)['type'].endswith('mvGroup') for v in children):
                    # Existing paired form columns (e.g. Batch temperature and
                    # unit) share the rail; neither may claim its full width.
                    columns*=len(children)
            self._constraint_cache[key]=(bound,columns)
            return bound,columns
        bound,columns=constraints(info(row.wrapper).get('parent'))
        if owned:
            width=getattr(self.app,'_owned_field_widths',{}).get(item,config(row.wrapper).get('width',0))
            if width>0:bound=min(bound,width)
        return max(1,int((bound-(columns-1)*TOKENS.spacing.item_spacing[0])/columns))

    def refresh(self):
        d=self.d;dead=[]
        # These are frame-local: docking/resizing/reparenting is observed on the
        # next frame, while shared ancestor queries are performed only once.
        self._info_cache={};self._config_cache={};self._state_cache={};self._constraint_cache={}
        for item,row in self.rows.items():
            if not all(d.does_item_exist(v) for v in (item,row.wrapper,row.heading)):
                dead.append(item);continue
            cfg=d.get_item_configuration(item);label=str(d.get_value(row.heading))
            width=self._available(item,row)
            measured=d.get_text_size(label,font=SECONDARY_FONT_TAG) if d.does_item_exist(SECONDARY_FONT_TAG) else d.get_text_size(label)
            if measured is None:continue
            kind=d.get_item_info(item)['type'].split('::')[-1]
            minimum=TOKENS.form.control_min
            if kind in ('mvInputDouble','mvInputFloat','mvInputInt') and cfg.get('step',0):minimum=TOKENS.form.stepped_control_min
            if kind=='mvCombo':
                value=d.get_text_size(str(d.get_value(item)))
                if value:minimum=max(minimum,min(280,value[0]+2*TOKENS.spacing.frame_padding[0]+TOKENS.typography.body_pt))
            inline,label_width,control_width=row_geometry(width,measured[0],minimum,
                short=row.scalar and '\n' not in label and len(label)<=TOKENS.form.label_max_chars)
            layout=(inline,label_width,control_width,label,width,cfg.get('show',True),measured,
                    getattr(self.app,'_display_layout_epoch',0))
            if self._layout_cache.get(item)==layout:continue
            self._layout_cache[item]=layout
            # DPG group.width is broadcast to descendants while drawing. Keep
            # it unset so it cannot overwrite the input and spacer widths.
            d.configure_item(row.wrapper,horizontal=inline,horizontal_spacing=0,width=0,show=cfg.get('show',True))
            # mvText has no width option in the project's DPG runtime. A real
            # spacer aligns the input column without rewriting label text.
            d.configure_item(row.spacer,show=inline,width=max(1,int(label_width-measured[0]+TOKENS.spacing.item_spacing[0])))
            d.configure_item(row.heading,wrap=0 if inline else width)
            d.configure_item(item,width=int(control_width))
            self.metrics[d.get_item_alias(item) or str(item)]=dict(label=label,inline=inline,available=width,
                control_width=control_width,label_width=label_width,scalar=row.scalar)
        for item in dead:
            self.labels.discard(self.rows[item].heading);del self.rows[item]
            self._layout_cache.pop(item,None)
