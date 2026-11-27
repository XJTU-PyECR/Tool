"""Versioned display transforms of captured objective grids, never objectives."""
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral
import numpy as np

VERSION = "error-map-display/2.0.0"
PALETTES = ("Blue Lavender Peach", "Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Coolwarm")

def _register_submission_palette():
    """Register the manuscript palette for both native and exported maps."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap
    name = 'pyecr_blue_lavender_peach'
    if name not in colormaps:
        cmap = LinearSegmentedColormap.from_list(
            name, ['#465D8A', '#737DA7', '#A49EC2', '#CFB6CF', '#F0C4B3', '#F0C4B3', '#EBA58D'])
        colormaps.register(cmap)
        colormaps.register(cmap.reversed(), name=name+'_r')
    return name
RESOLUTIONS = {"Quick": 41, "Standard": 81, "Fine": 151, "High 201": 201, "Ultra 301": 301}

@dataclass(frozen=True)
class MapViewOptions:
    metric: str = "log10(SSE/SSE_min)"
    shared_scale: bool = False
    show_grid_min: bool = True
    palette: str = "Blue Lavender Peach"
    reverse: bool = False
    sampling: str = "Cells"
    contrast: str = "Full"
    color_range: tuple[float, float] | None = None
    padding: float = .04
    raster_points: int = 501
    image_dpi: int = 300
    display_ranges: tuple = ()  # (process, log10 D min/max, log10 k min/max)
    range_mode: str = 'Auto'

    def __post_init__(self):
        if self.range_mode not in ('Auto','Custom'):raise ValueError('range_mode must be Auto or Custom')
        from pyecr.plotting.map_ranges import log_range
        entries=[]
        for process,d,k in self.display_ranges:
            if process not in ('single','H','O') or process in [e[0] for e in entries]:
                raise ValueError('unique single/H/O view ranges required')
            entries.append((process,log_range(d,coordinate='log10'),log_range(k,coordinate='log10')))
        object.__setattr__(self,'display_ranges',tuple(sorted(entries)))
        if self.metric not in ("log10(SSE/SSE_min)","SSE/SSE_min","SSE"):raise ValueError("unknown map metric")
        if not isinstance(self.shared_scale,bool) or not isinstance(self.show_grid_min,bool):raise TypeError("map switches must be boolean")
        if self.palette not in PALETTES: raise ValueError("unknown map palette")
        if not isinstance(self.reverse, bool): raise TypeError("reverse must be boolean")
        if self.sampling not in ("Cells", "Smooth"): raise ValueError("sampling must be Cells or Smooth")
        if self.contrast not in ("Full", "P95", "P99"): raise ValueError("contrast must be Full, P95 or P99")
        if not isfinite(self.padding) or not 0 <= self.padding <= .5: raise ValueError("axis padding must be 0–0.5")
        if isinstance(self.raster_points, bool) or not isinstance(self.raster_points, Integral) or not 3 <= self.raster_points <= 1001:
            raise ValueError("display raster must have 3–1001 points")
        if isinstance(self.image_dpi,bool) or not isinstance(self.image_dpi,Integral) or not 72<=self.image_dpi<=1200:
            raise ValueError("map PNG DPI must be integer 72–1200")
        if self.color_range is not None:
            a,b=self.color_range
            if not all(isfinite(x) for x in (a,b)) or a >= b: raise ValueError("color min,max must be finite and increasing")
            object.__setattr__(self,"color_range",(float(a),float(b)))

    @property
    def cmap_name(self):
        name = _register_submission_palette() if self.palette == 'Blue Lavender Peach' else self.palette.lower()
        return name + ("_r" if self.reverse else "")
    def ranges(self,process='single'):
        return next(((d,k) for p,d,k in self.display_ranges if p==process),None)
    def raster_options(self):
        from dataclasses import replace
        return replace(self,display_ranges=(),range_mode='Auto')
    def to_dict(self): return {"version":VERSION,**asdict(self)}
    @classmethod
    def from_dict(cls,value):
        if value.get("version",VERSION) not in ("error-map-display/1.0.0",VERSION): raise ValueError("unsupported Error Map view version")
        return cls(**{k:v for k,v in value.items() if k!="version"})

def color_limits(values, mask, options):
    finite=np.asarray(values)[np.asarray(mask) & np.isfinite(values)]
    if not finite.size: return None
    if options.color_range is not None:return options.color_range
    lo=float(np.min(finite));hi=float(np.max(finite) if options.contrast=="Full" else np.percentile(finite,int(options.contrast[1:])))
    return lo,hi if hi>lo else lo+max(abs(lo),1.)*1e-12

def display_raster(values, mask, options, *, color_range=None):
    """D/k row order. Bilinear display pixels never replace stored cells.

    Invalid cells are not filled: every contributing corner must be valid.
    Bilinear convex weights cannot create a value outside source extrema.
    """
    source=np.asarray(values,dtype=float);valid=np.asarray(mask,dtype=bool)&np.isfinite(source)
    limits=color_range or color_limits(source,valid,options)
    if limits is not None:source=np.clip(source,*limits)
    if options.sampling=="Cells":
        shown=np.where(valid,source,np.nan)
    else:
        n=options.raster_points
        def coords(size):
            x=np.clip((np.arange(n)+.5)*size/n-.5,0,size-1)
            i=np.floor(x).astype(int);return i,np.minimum(i+1,size-1),x-i
        a,b,u=coords(source.shape[0]);c,d,v=coords(source.shape[1])
        shown=np.zeros((n,n));mass=np.zeros_like(shown)
        safe=np.where(valid,source,0.)
        for i,j,weight in ((a,c,(1-u)[:,None]*(1-v)),(a,d,(1-u)[:,None]*v),
                           (b,c,u[:,None]*(1-v)),(b,d,u[:,None]*v)):
            shown+=weight*safe[np.ix_(i,j)];mass+=weight*valid[np.ix_(i,j)]
        shown[mass<1-1e-12]=np.nan
    shown.flags.writeable=False;return shown

def padded_limits(edges, options):
    lo,hi=float(edges[0]),float(edges[-1]);gap=(hi-lo)*options.padding
    return lo-gap,hi+gap

def grid_points(label, custom=201):
    size=custom if label=="Custom" else RESOLUTIONS.get(label)
    if isinstance(size,bool) or not isinstance(size,Integral) or not 3<=size<=401:
        raise ValueError("calculated grid points must be integer 3–401 per axis")
    return int(size)

def palette_colors(name):
    """One 256-color RGBA table for native and publication renderers."""
    from matplotlib import colormaps
    _register_submission_palette()
    if name in ('Blue Lavender Peach', 'blue lavender peach', 'blue lavender peach_r'):
        name = 'pyecr_blue_lavender_peach' + ('_r' if name.endswith('_r') else '')
    return np.rint(colormaps[name](np.linspace(0,1,256))*255).astype(np.uint8)

def snapshot_map_options(snapshot,family):
    metadata=snapshot.export_context.get("project_metadata",{})
    value=metadata.get("error_map_view",{}).get(family)
    return None if value is None else MapViewOptions.from_dict(value)
