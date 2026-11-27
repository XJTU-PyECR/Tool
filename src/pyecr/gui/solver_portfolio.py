"""Presentation of core portfolio records; no objective/solver computation."""
from pyecr.fitting.solver_strategy import SolverOptions


def config_selection(strategy,policy='warn',existing=None):
    result=dict(existing or {})
    limit=result.get('local_solver',{}).get('condition_limit',1e8)
    result.pop('local_solver',None)
    if strategy!='trf':
        options=SolverOptions(strategy,policy,limit)
        result['local_solver']=dict(strategy=options.strategy,dogbox_policy=options.dogbox_policy,
                                    condition_limit=options.condition_limit)
    return result


def solver_details(runs,fit=None):
    if not runs:
        meta={} if fit is None else fit.diagnostics.get('local_solver',{})
        return '\n'.join(('Solver: '+str(meta.get('strategy','legacy TRF')),
                'Transient solver counters unavailable: no live fit, loaded project, or artifact cache expired.',
                *meta.get('warnings',())))
    lines=[]
    for run in runs:
        r=run.to_dict() if hasattr(run,'to_dict') else run
        lines.append(f"{r['strategy']} {r['model_id']} start {r['start_index']+1}: {r['status']} | candidate rank={r.get('candidate_rank')} | nfev={r.get('nfev')} njev={r.get('njev')}")
        lines.append(f"residual calls={r['actual_residual_calls']} (numeric Jacobian included); diagnostic calls={r['diagnostic_residual_calls']}; wall={r['wall_s']:.3f}s; RSS={r['RSS_after']['RSS_bytes']}")
        lines.append(f"full-data SSE={r.get('full_data_SSE')} [{r['full_data_scope']}]; optimality={r.get('optimality')} [{r.get('optimality_coordinate_space')}]")
        lines.append('Termination: '+r.get('termination_reason','unavailable'))
        lines.extend(r.get('warnings',()))
        if r.get('njev_reason'):lines.append(r['njev_reason'])
        if r.get('unavailable_reason'):lines.append('Unavailable fields: '+r['unavailable_reason'])
        if r['RSS_after'].get('reason'):lines.append('RSS unavailable: '+r['RSS_after']['reason'])
    return '\n'.join(lines)
