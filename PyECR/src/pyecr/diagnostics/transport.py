"""Unique full-size/half-size transport calculation, shared with old Bi diagnostics."""
from math import isfinite
from numbers import Real
from math import log
from pyecr.config import SampleGeometry
from pyecr.models.ids import ModelID

VERSION='transport-regime/1'
DEFINITIONS={
    ModelID.M0_SINGLE_D:(('single','Single process','D_cm2_s',None),),
    ModelID.M1_SINGLE_DK:(('single','Single process','D_cm2_s','k_cm_s'),),
    ModelID.M2_DUAL_HO_D:(('H','Parent role H','D_H_cm2_s',None),('O','Parent role O','D_O_cm2_s',None)),
    ModelID.M3_DUAL_HO_DK:(('H','Parent role H','D_H_cm2_s','k_H_cm_s'),('O','Parent role O','D_O_cm2_s','k_O_cm_s')),
    ModelID.M7_SINGLE_K:(('single','Single effective process',None,'k_cm_s'),),
    ModelID.M8_DUAL_K:(('component_1','Effective mode 1',None,'k_1_cm_s'),('component_2','Effective mode 2',None,'k_2_cm_s'))}


def positive(value):
    return not isinstance(value,bool) and isinstance(value,Real) and isfinite(value) and value>0


def point_values(D,k,geometry,surface_limit=.03,diffusion_limit=30.):
    """Complete input L in cm. Return D/k, 2D/k and active-axis tuples.

    Invalid/missing kinetic values give N/A, never infinity or an inferred D.
    The full-thickness reference is specifically a symmetric two-face slab;
    multi-axis specimens retain separate Bi per active half-dimension.
    """
    if not isinstance(geometry,SampleGeometry):raise TypeError('Expected full SampleGeometry')
    geometry.validate()
    if not(positive(surface_limit) and positive(diffusion_limit) and surface_limit<diffusion_limit):raise ValueError('Invalid heuristic limits')
    half=full=None
    if positive(D) and positive(k):
        half=float(D/k);full=2.*half
        if not(positive(half) and positive(full)):half=full=None
    axes=[]
    for axis,L,active in zip('xyz',geometry.full_dims_cm,geometry.active_dimensions,strict=True):
        if not active:continue
        value=None if half is None else float((L/2.)*k/D)
        if not positive(value):value=None
        regime='unavailable' if value is None else 'surface_limited' if value<surface_limit else 'diffusion_limited' if value>diffusion_limit else 'mixed_control'
        axes.append((axis,float(L),value,regime))
    return half,full,tuple(axes)


def transport_points(model_id,geometry,parameters,*,fixed=(),given=False,labels=None,surface_limit=.03,diffusion_limit=30.):
    from pyecr.contracts.transport import TransportProcess,TransportAxis
    if model_id not in DEFINITIONS:raise ValueError('Unsupported transport model_id')
    processes=[];axes=[]
    for process,label,dn,kn in DEFINITIONS[model_id]:
        D=parameters.get(dn) if dn else None;k=parameters.get(kn) if kn else None
        reasons=[]
        for name,value,role in ((dn,D,'D'),(kn,k,'k')):
            if name is None:reasons.append(role+'_not_in_model')
            elif name not in parameters:reasons.append('Missing '+name)
            elif not positive(value):reasons.append(name+' must be positive and finite')
        half,full,points=point_values(D,k,geometry,surface_limit,diffusion_limit)
        if half is None and not reasons:reasons.append('Transport scale overflow/underflow')
        why='; '.join(reasons)
        origin=lambda n:'not_in_model' if n is None else 'given' if given else 'fixed' if n in fixed else 'estimated'
        processes.append(TransportProcess(process=process,label=(labels or {}).get(process,label),D_name=dn,k_name=kn,
            D_cm2_s=float(D) if positive(D) else None,k_cm_s=float(k) if positive(k) else None,D_origin=origin(dn),k_origin=origin(kn),
            half_length_cm=half,slab_full_thickness_cm=full,reason=why))
        axes.extend(TransportAxis(process=process,axis=a,full_dimension_cm=L,value=b,regime=r,
            reason=why or ('Axis Bi overflow/underflow' if b is None else '')) for a,L,b,r in points)
    return tuple(processes),tuple(axes)


def log_ratio_gradient(order,coordinates,parameters,D_name,k_name):
    """d ln(D/k) / d each actual parent coordinate; fixed slots have no column."""
    result=[0.]*len(order)
    for name,sign in ((D_name,1.),(k_name,-1.)):
        if name not in order:continue
        space=coordinates[name]
        if space not in ('log','linear'):raise ValueError('Unsupported parent coordinate: log means log10, not ln')
        if not positive(parameters[name]):raise ValueError('Finite positive parameter required for ratio gradient')
        result[order.index(name)]=sign*(log(10.) if space=='log' else 1./parameters[name])
    return tuple(result)


def local_interval_bounds(value,variance,C,df):
    """One local delta interval in log-ratio space; no marginal CI endpoints."""
    from math import exp,sqrt
    from pyecr.uncertainty.local import student_t
    if not(positive(value) and isfinite(variance) and variance>=0 and 0<C<1 and df>0):raise ValueError('Invalid derived interval evidence')
    h=float(student_t.ppf((1+C)/2,df))*sqrt(variance)
    low,high=value*exp(-h),value*exp(h)
    if not(positive(low) and positive(high)):raise ValueError('Derived interval overflow/underflow')
    return low,high
