"""DCT client-space geometry; it never consumes scientific arrays or config."""
from dataclasses import dataclass
from pyecr.gui.theme import TOKENS

VERSION='dct-layout/2.0.0'


@dataclass(frozen=True,slots=True)
class DctGeometry:
    left: int
    right: int
    panel_height: int
    spectrum_width: int
    spectrum_height: int
    reconstruction_width: int
    reconstruction_height: int
    residual_width: int
    residual_height: int


def geometry(client_width,available_height):
    spec=TOKENS.dct
    width=int(client_width);height=max(spec.panel_min,int(available_height))
    left=spec.left_small if width<1200 else spec.left_compact if width<1600 else spec.left_medium if width<2200 else spec.left_large
    right=max(spec.panel_min,width-spec.margin-spec.gap-left)
    main=max(spec.spectrum_min,min(spec.spectrum_max,int(height*spec.spectrum_fraction)))
    reconstruction=max(spec.reconstruction_min,min(spec.reconstruction_max,int(height*.58)))
    # Residuals has its own tab, unlike the two stacked reconstruction/spectrum
    # plots. Use that tab's client space rather than the old 30% inset.
    residual=max(spec.residual_min,min(spec.residual_max,height-spec.residual_chrome))
    plot_width=lambda h:min(right-spec.plot_gutter,int(h*spec.aspect_max))
    return DctGeometry(left,right,height,plot_width(main),main,plot_width(reconstruction),
                       reconstruction,right-spec.plot_gutter,residual)
