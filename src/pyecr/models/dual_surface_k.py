"""Two independent effective surface-exchange modes, sharing the P68 kernel."""
from dataclasses import dataclass
import numpy as np
from pyecr.models.surface_k import SurfaceKKernel, finite_scalar
from pyecr.models.ids import ModelID

@dataclass(frozen=True, slots=True)
class DualSurfaceKModel:
    kernel: SurfaceKKernel
    signs: tuple[float,float] = (-1.,1.)
    model_id: ModelID = ModelID.M8_DUAL_K

    def __post_init__(self):
        if not isinstance(self.kernel,SurfaceKKernel) or self.model_id is not ModelID.M8_DUAL_K:
            raise TypeError("M8 requires the independent P68 SurfaceKKernel")
        if len(self.signs)!=2 or any(s not in (-1.,1.) or isinstance(s,bool) for s in self.signs):
            raise ValueError("M8 input signs must each be +/-1; zero process uses A=0")

    def evaluate(self,t,params):
        unknown=set(params)-{"k_1_cm_s","k_2_cm_s","A_1","A_2","y0"}
        if unknown: raise ValueError(f"M8 undeclared parameters: {sorted(unknown)}")
        y0=finite_scalar(params.get("y0",0.),"y0")
        components=[]
        for slot,sign in enumerate(self.signs,1):
            a=finite_scalar(params[f"A_{slot}"],f"A_{slot}")
            if a<0: raise ValueError("M8 amplitudes must be nonnegative")
            x,_=self.kernel.evaluate(t,params[f"k_{slot}_cm_s"])
            components.append(sign*a*x)
        total=y0+components[0]+components[1]
        if not np.all(np.isfinite(total)): raise ValueError("M8 nonfinite total signal")
        total.flags.writeable=False
        for component in components: component.flags.writeable=False
        return total,tuple(components)

    def predict(self,t,params,context=None):
        return self.evaluate(t,params)[0]

    def jacobian(self,t,params,names):
        columns={"y0":np.ones(np.asarray(t).shape)}
        for slot,sign in enumerate(self.signs,1):
            x,dk=self.kernel.evaluate(t,params[f"k_{slot}_cm_s"])
            a=finite_scalar(params[f"A_{slot}"],f"A_{slot}")
            if a<0: raise ValueError("M8 amplitudes must be nonnegative")
            columns[f"k_{slot}_cm_s"]=sign*a*dk
            columns[f"A_{slot}"]=sign*x
        return np.column_stack([columns[name] for name in names])

def input_signs(config):
    if config.step_type=="hydration": return (-1.,1.)
    if config.step_type=="dehydration": return (1.,-1.)
    return (float(config.sign_H),float(config.sign_O))
