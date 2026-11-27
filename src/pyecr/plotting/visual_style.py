"""Versioned visual roles shared by native and publication renderers only."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

VERSION='pyecr-visual/1.1.0'


@dataclass(frozen=True)
class SeriesStyle:
    hex: str
    line: str
    width: float
    marker_size: float

    @property
    def rgba(self):
        return tuple(int(self.hex[i:i+2],16) for i in (1,3,5))+(255,)


SERIES=MappingProxyType({
    'observed':SeriesStyle('#16858d','none',1.,12.),
    'total':SeriesStyle('#ef7d24','-',2.,0.),
    'one':SeriesStyle('#c74747','--',1.5,0.),
    'two':SeriesStyle('#245fa6','--',1.5,0.),
    'reference':SeriesStyle('#777777',':',.9,0.),
    'zero':SeriesStyle('#777777','-',.9,0.),
    'residual':SeriesStyle('#777777','none',1.,10.),
})


@dataclass(frozen=True)
class PublicationStyle:
    body_pt: float=10.
    label_pt: float=11.
    title_pt: float=12.
    tick_pt: float=10.
    legend_pt: float=9.5
    note_pt: float=9.
    inset_tick_pt: float=8.
    face: str='white'
    grid_alpha: float=.15
    footer_fraction: float=.015
    inset_bounds: tuple[float,float,float,float]=(.55,.16,.41,.32)
    residual_height_ratios: tuple[float,float]=(3.,1.25)


PUBLICATION=PublicationStyle()


def marker_indices(times,signal,budget=300):
    """Display positions only: t0/ends, early rows and bin extrema; no data edits."""
    t=np.asarray(times);y=np.asarray(signal);n=len(t)
    if t.ndim!=1 or y.shape!=t.shape or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError('finite aligned display vectors required')
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<24:raise ValueError('display budget must be integer >=24')
    if n<=budget:return np.arange(n,dtype=int)
    chosen={0,n-1,int(np.argmin(y)),int(np.argmax(y)),*map(int,np.flatnonzero(t==0))}
    chosen.update(range(min(16,n)))
    bins=max(1,(budget-len(chosen))//2)
    for group in np.array_split(np.arange(n),bins):
        if len(group):chosen.update((int(group[np.argmin(y[group])]),int(group[np.argmax(y[group])])) )
    return np.array(sorted(chosen),dtype=int)


def native_line_geometry(times,values,style='-',segments=60):
    """NaN-separated dash geometry for ImPlot, never an observation/fit grid.

    Finite dash vertices are linear interpolation of the authoritative displayed
    polyline; no residuals, statistics, snapshots or CSV consume this geometry.
    """
    x=np.asarray(times,dtype=float);y=np.asarray(values,dtype=float)
    if style=='-' or len(x)<2:return x.tolist(),y.tolist()
    if x.ndim!=1 or x.shape!=y.shape or np.any(np.diff(x)<=0):raise ValueError('ordered display polyline required')
    edges=np.linspace(x[0],x[-1],segments+1);fraction=.55 if style=='--' else .15
    xx=[];yy=[]
    for a,b in zip(edges[:-1],edges[1:]):
        end=a+(b-a)*fraction
        inner=x[(x>a)&(x<end)]
        pts=np.r_[a,inner,end]
        xx.extend(pts.tolist()+[float('nan')]);yy.extend(np.interp(pts,x,y).tolist()+[float('nan')])
    return xx,yy


def publication_fonts(preferred='DejaVu Sans'):
    """Reference checked existing fonts; Matplotlib falls back per glyph."""
    from matplotlib import font_manager
    from pyecr.gui.fonts import choose_fonts
    plan=choose_fonts();names=[preferred]
    for face in (plan.body,plan.scientific):
        if face is not None:
            font_manager.fontManager.addfont(str(face.path))
            if face.family not in names:names.append(face.family)
    return names,plan


def style_publication_figure(figure,preferred='DejaVu Sans'):
    """Typography/white canvas only. Arrays and axis limits stay untouched."""
    from matplotlib.text import Text
    fonts,plan=publication_fonts(preferred)
    figure.set_facecolor(PUBLICATION.face)
    for text in figure.findobj(Text):text.set_fontfamily(fonts)
    for axis in figure.axes:
        axis.set_facecolor(PUBLICATION.face)
        axis.xaxis.label.set_fontsize(PUBLICATION.label_pt);axis.yaxis.label.set_fontsize(PUBLICATION.label_pt)
        axis.title.set_fontsize(PUBLICATION.title_pt);axis.tick_params(labelsize=PUBLICATION.tick_pt)
        legend=axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():text.set_fontsize(PUBLICATION.legend_pt)
    return plan
