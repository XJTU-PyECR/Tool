"""Versioned presentation only. No dependency on models, tasks or scientific hashes."""
from dataclasses import dataclass, replace
import json
from math import isfinite
from collections.abc import Mapping

VERSION = 2
METADATA_KEY = 'plot_views'
POSITIONS = {'top_left':(0.,0.), 'top':(.5,0.), 'top_right':(1.,0.),
             'left':(0.,.5), 'center':(.5,.5), 'right':(1.,.5),
             'bottom_left':(0.,1.), 'bottom':(.5,1.), 'bottom_right':(1.,1.)}
SIDES = ('top','bottom','left','right')
V1_FIELDS = {'legend_anchor','legend_visible','hidden_series','x_range','y_range','hidden_layers'}


def _pair(value, *, bounds=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('view coordinate must contain two numbers')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not isfinite(x) for x in value):
        raise ValueError('view coordinate must be finite')
    a, b = map(float, value)
    if bounds and a >= b:
        raise ValueError('view limits must increase')
    return a, b


@dataclass(frozen=True, slots=True)
class PlotViewState:
    # Top-left legend corner, relative to the INNER plot, y increasing downwards.
    legend_anchor: tuple[float, float] = (.02, .02)
    legend_visible: bool = True
    hidden_series: tuple[str, ...] = ()
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    hidden_layers: tuple[str, ...] = ()
    legend_mode: str = 'free'
    legend_position: str = 'top_left'
    outside_side: str = 'right'
    outside_align: float = .5
    legend_frame: bool = True

    def __post_init__(self):
        a = _pair(self.legend_anchor)
        object.__setattr__(self, 'legend_anchor', tuple(max(0., min(1., x)) for x in a))
        if type(self.legend_visible) is not bool:
            raise ValueError('legend visibility must be boolean')
        if type(self.legend_frame) is not bool:
            raise ValueError('legend frame must be boolean')
        if self.legend_mode not in ('free','preset','outside') or self.legend_position not in POSITIONS or self.outside_side not in SIDES:
            raise ValueError('unsupported legend placement')
        if isinstance(self.outside_align,bool) or not isinstance(self.outside_align,(int,float)) or not isfinite(self.outside_align) or not 0<=self.outside_align<=1:
            raise ValueError('outside alignment must be between zero and one')
        for name in ('hidden_series', 'hidden_layers'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError('view series/layer IDs must be nonempty strings')
            object.__setattr__(self, name, tuple(sorted(set(values))))
        for name in ('x_range', 'y_range'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _pair(getattr(self, name), bounds=True))

    def to_dict(self):
        return dict(legend_anchor=list(self.legend_anchor), legend_visible=self.legend_visible,
                    hidden_series=list(self.hidden_series), x_range=self.x_range,
                    y_range=self.y_range, hidden_layers=list(self.hidden_layers),
                    legend_mode=self.legend_mode,legend_position=self.legend_position,
                    outside_side=self.outside_side,outside_align=self.outside_align,legend_frame=self.legend_frame)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('invalid plot view fields')
        return cls(**value)

    def reset_legend(self):
        defaults = default_plot_view()
        return replace(self, **{name: getattr(defaults, name) for name in
            ('legend_anchor', 'legend_visible', 'legend_mode', 'legend_position',
             'outside_side', 'outside_align', 'legend_frame')})


def default_plot_view():
    """New charts use inside-right-center; saved v1/v2 defaults retain their wire."""
    return PlotViewState(legend_anchor=(.98, .5), legend_mode='preset', legend_position='right')


def decode_views(value=None):
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping) or set(value) != {'version', 'plots'} or type(value['version']) is not int or value['version'] not in (1,VERSION):
        raise ValueError('unsupported plot view version')
    if not isinstance(value['plots'], Mapping) or any(not isinstance(k, str) or not k for k in value['plots']):
        raise ValueError('stable plot IDs required')
    if value['version']==1 and any(not isinstance(v,Mapping) or set(v)-V1_FIELDS for v in value['plots'].values()):
        raise ValueError('v1 plot view cannot contain new placement fields')
    return {k: PlotViewState.from_dict(v) for k, v in value['plots'].items()}


def encode_views(views, *, version=VERSION):
    """GUI edits use v2; immutable legacy adapters may retain their v1 wire."""
    if type(version) is not int or version not in (1,VERSION):
        raise ValueError('unsupported plot view version')
    if version==1:
        defaults=PlotViewState().to_dict()
        for state in views.values():
            if any(value!=defaults[name] for name,value in state.to_dict().items() if name not in V1_FIELDS):
                raise ValueError('cannot discard v2 legend placement in v1')
    return dict(version=version, plots={k:{n:value for n,value in v.to_dict().items() if version!=1 or n in V1_FIELDS}
                                       for k, v in sorted(views.items())})


def freeze_views(value=None):
    """A JSON string is an immutable, bounded display snapshot for worker exports."""
    raw=json.loads(value) if isinstance(value,str) else value
    views=decode_views(raw)
    # Read-only legacy packages bind the original v1 wire in their catalogs.
    # Upgrade when a user edits (encode_views), not while verifying old hashes.
    wire=encode_views(views,version=1 if raw is not None and raw['version']==1 else VERSION)
    return json.dumps(wire, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def legend_position(rect, size, anchor):
    """Clamp the entire legend inside the plot; oversized labels remain grab-able."""
    x, y, width, height = rect
    w, h = size
    return (x + min(max(0., width-w), max(0., anchor[0]*width)),
            y + min(max(0., height-h), max(0., anchor[1]*height)))


def placed_legend(rect,size,state):
    """Inside presets align the full box; free anchors retain continuous v1 semantics."""
    if state.legend_mode!='preset':return legend_position(rect,size,state.legend_anchor)
    a,b=POSITIONS[state.legend_position];x,y,w,h=rect
    return (x+max(0.,w-size[0])*a,y+max(0.,h-size[1])*b)


def export_legend_position(rect, size, state, dpi):
    """Keep complete legends inside across raster/vector font rounding."""
    x, y, w, h = rect
    padding = dpi * 6 / 72
    px, py = min(padding, max(0., (w-size[0])/2)), min(padding, max(0., (h-size[1])/2))
    if state.legend_mode != 'preset':
        left, top = placed_legend(rect, size, state)
        return max(x+px, min(x+w-size[0]-px, left)), max(y+py, min(y+h-size[1]-py, top))
    return placed_legend((x+px, y+py, w-2*px, h-2*py), size, state)


def wrap_label(label,width,measure):
    """Presentation line breaks only; retain source text and support CJK/no spaces."""
    lines=[]
    for source in label.split('\n'):
        line=''
        for char in source:
            if line and measure(line+char)>width:lines.append(line);line=char
            else:line+=char
        lines.append(line)
    return '\n'.join(lines)


def apply_axis_view(axis, plot_id, frozen, series_ids=(), *, log_coordinates=False, ranges=True, legend_policy='follow'):
    """Consume frozen state after layout, before save; IDs never depend on user labels."""
    if legend_policy not in ('follow', 'auto'):
        raise ValueError('unsupported export legend policy')
    state = decode_views(frozen).get(plot_id,default_plot_view())
    flags=(True,True) if log_coordinates is True else log_coordinates if isinstance(log_coordinates,tuple) else (False,False)
    state = replace(state, x_range=None if state.x_range is None else tuple(10**v for v in state.x_range) if flags[0] else state.x_range,
                    y_range=None if state.y_range is None else tuple(10**v for v in state.y_range) if flags[1] else state.y_range)
    handles, labels = axis.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection
    bars={id(p):c for c in axis.containers if hasattr(c,'patches') for p in c.patches}
    artists=[]
    for a in axis.get_children():
        if isinstance(a,(Line2D,PathCollection)):artists.append(a)
        elif id(a) in bars and not any(bars[id(a)] is v for v in artists):artists.append(bars[id(a)])
    def visible(a):return a.get_visible() if hasattr(a,'get_visible') else any(p.get_visible() for p in a.patches)
    def hide(a):
        if hasattr(a,'set_visible'):a.set_visible(False)
        else:
            for p in a.patches:p.set_visible(False)
    for handle, key in zip(artists, series_ids):
        if key in state.hidden_series or key in state.hidden_layers:
            hide(handle)
    if ranges and state.x_range is not None:
        axis.set_xlim(*state.x_range)
    if ranges and state.y_range is not None:
        axis.set_ylim(*state.y_range)
    legend = axis.get_legend()
    if not state.legend_visible:
        if legend is not None:
            legend.set_visible(False)
        return
    if not handles:
        return
    if legend is None:
        legend = axis.legend()
    legend.set_visible(True)
    for handle, artist in zip(legend.legend_handles, handles):
        if not visible(artist):
            handle.set_alpha(.3)
    # Clamp using final axes/text dimensions, not screen coordinates. Draw once to
    # resolve publication constrained/tight layout, then place in axes coordinates.
    from .legend_layout import place_export_legend
    place_export_legend(axis,legend,state,policy=legend_policy)

