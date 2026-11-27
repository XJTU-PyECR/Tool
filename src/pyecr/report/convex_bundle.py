"""Strict, independent frozen experimental evidence. Read/export never solves."""
from pathlib import Path
from hashlib import sha256
import csv,html,json
import numpy as np
from pyecr.contracts import DctSourceSnapshot,SignedDctConfig
from pyecr.contracts.common import require

KIND='signed-convex-evidence/1'


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def validate_report(report):
    from pyecr.dct.signed_convex import ConvexSpec,whiten
    require(isinstance(report,dict) and report.get('version')=='signed-convex-search/1','unsupported convex search version')
    source=DctSourceSnapshot.from_dict(report['source']);source.require_usable()
    cfg=SignedDctConfig.from_dict(report['config']);spec=ConvexSpec(**report['spec'])
    require(report.get('experimental') is True and report.get('default_changed') is False,'experimental/current contract conflict')
    require(report.get('strategy')=='known_noise_discrepancy' and bool(report.get('noise_evidence')),'missing noise evidence')
    trials=report['trials'];require(isinstance(trials,list) and 1<=len(trials)<=report['max_candidates'],'invalid candidate budget')
    y=np.asarray(source.signal);time=np.asarray(source.time_s)
    weights=np.asarray(cfg.observation_weights or np.ones(len(y)))
    sd=np.asarray(report['noise_sd_physical'])
    require(sd.shape==y.shape and np.all(np.isfinite(sd)) and np.all(sd>0),'invalid declared noise')
    adjusted=sd*np.sqrt(weights)/source.normalization.scale_value
    require(np.allclose(adjusted,adjusted[0],rtol=1e-6,atol=0),'precision/noise mismatch')
    require(np.isclose(report['target_sse'],len(y)*adjusted[0]**2,rtol=1e-12,atol=0),'discrepancy target mismatch')
    lambdas=[]
    for trial in trials:
        require(trial['version']==spec.version and trial['spec']==report['spec'] and trial['source_sha256']==source.content_sha256,'candidate provenance mismatch')
        lam=trial['lambda_value'];require(np.isfinite(lam) and lam>0,'invalid lambda');lambdas.append(lam)
        require(trial['fallback'] is None and trial['df'] is None and trial['gcv'] is None,'unexpected fallback/nonlinear GCV')
        if trial['status']!='CONVERGED':
            require(trial['status'].startswith('FAIL_') and trial['chi'] is None and trial['reconstruction'] is None,'failed candidate has main spectrum')
            continue
        tau=np.asarray(trial['tau_s']);q=np.asarray(trial['quadrature_weights']);chi=np.asarray(trial['chi']);mass=np.asarray(trial['quadrature_mass'])
        require(tau.shape==q.shape==chi.shape==mass.shape==(cfg.grid.n_tau,),'spectrum shape mismatch')
        require(np.all(np.isfinite(chi)) and np.all(q>0) and np.all(np.diff(tau)>0),'invalid spectrum arrays')
        x=np.log10(tau);h=(x[-1]-x[0])/(len(x)-1);expected_q=np.full(len(q),h);expected_q[[0,-1]]/=2
        require(np.allclose(x,np.linspace(np.log10(cfg.grid.tau_min_s),np.log10(cfg.grid.tau_max_s),len(x)),rtol=0,atol=1e-13),'grid conflict')
        require(np.allclose(q,expected_q,rtol=1e-12,atol=1e-14) and np.allclose(mass,q*chi,rtol=1e-12,atol=1e-13),'density/mass/quadrature conflict')
        prediction=np.empty(len(time))
        for i in range(0,len(time),4096):prediction[i:i+4096]=np.exp(-time[i:i+4096,None]/tau)@mass
        require(np.allclose(prediction,trial['reconstruction'],rtol=1e-12,atol=1e-12) and np.allclose(y-prediction,trial['residuals'],rtol=1e-12,atol=1e-12),'reconstruction/residual mismatch')
        residual=whiten(np.sqrt(weights)*(y-prediction),time,spec.ou_correlation_seconds)
        sse=float(residual@residual);alpha=spec.mass_ratio*np.sqrt(lam)
        curvature=float(lam*np.sum((np.diff(chi,2)*np.sqrt(h)/h**2)**2))
        distance=np.minimum(x-x[0],x[-1]-x);inside=distance<spec.edge_width_decades
        edge_weights=np.where(inside,.5*(1+np.cos(np.pi*distance/spec.edge_width_decades)),0.)
        penalties={'weighted_sse':sse,'curvature_penalty':curvature,'absolute_mass_penalty':float(alpha*np.abs(mass).sum()),
            'edge_penalty':float(lam*spec.edge_beta*np.sum(q*edge_weights*chi**2))}
        for name,v in penalties.items():require(np.isclose(v,trial[name],rtol=2e-9,atol=1e-11),'objective component mismatch: '+name)
        require(np.isclose(sum(penalties.values()),trial['objective'],rtol=2e-9,atol=1e-11),'objective mismatch')
        score=abs(np.log(max(sse,np.finfo(float).tiny)/report['target_sse']))
        require(np.isclose(score,trial['selection_score'],rtol=2e-8,atol=1e-9),'score mismatch')
        diag=trial['diagnostics']
        for name,v in {'positive_mass':np.maximum(mass,0).sum(),'negative_mass_magnitude':np.maximum(-mass,0).sum(),
            'absolute_mass':np.abs(mass).sum(),'signed_mass':mass.sum()}.items():
            require(np.isclose(v,diag[name],rtol=1e-12,atol=1e-12),'diagnostic mass mismatch')
        require(trial['iterations']<=spec.max_iterations and trial['kkt']<=spec.kkt_tolerance,'convergence/budget mismatch')
    require(len(set(lambdas))==len(lambdas),'duplicate candidate lambda')
    good=[r for r in trials if r['status']=='CONVERGED']
    selected=report['selected_lambda']
    if not good:require(selected is None and not report['tie_lambdas'],'failed search selection conflict')
    else:
        best=min(r['selection_score'] for r in good)
        ties=[r['lambda_value'] for r in good if np.isclose(r['selection_score'],best,rtol=1e-10,atol=1e-18)]
        require(set(ties)==set(report['tie_lambdas']) and selected==max(ties),'selected lambda/tie mismatch')
    return source


def read_convex_evidence(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    require(set(payload)=={'kind','report','metadata','report_sha256'},'unknown evidence fields')
    require(payload['kind']==KIND,'unsupported convex evidence version')
    require(sha256(canonical(payload['report'])).hexdigest()==payload['report_sha256'],'report hash mismatch')
    validate_report(payload['report']);return payload


def export_convex_bundle(report,directory,*,metadata):
    source=validate_report(report);path=Path(directory)
    require(not path.exists(),'refuse to overwrite experimental evidence')
    payload=dict(kind=KIND,report=report,metadata=metadata,report_sha256=sha256(canonical(report)).hexdigest())
    path.mkdir(parents=True)
    (path/'evidence.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    read_convex_evidence(path/'evidence.json')
    selected=next((r for r in report['trials'] if r['lambda_value']==report['selected_lambda']),None)
    with (path/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['lambda_value','status','selection_score','weighted_sse','curvature_penalty','absolute_mass_penalty','edge_penalty','iterations','seconds','kkt']
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['trials'])
    if selected:
        with (path/'spectrum.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['tau_s','chi_per_decade','signed_mass','physical_chi','physical_signed_mass'])
            s=source.normalization.scale_value
            writer.writerows([[format(v,'.17g') for v in row] for row in zip(selected['tau_s'],selected['chi'],selected['quadrature_mass'],np.array(selected['chi'])*s,np.array(selected['quadrature_mass'])*s)])
        with (path/'reconstruction.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['time_s','Y','reconstruction','residual'])
            writer.writerows([[format(v,'.17g') for v in row] for row in zip(source.time_s,source.signal,selected['reconstruction'],selected['residuals'])])
    body=html.escape(json.dumps(dict(metadata=metadata,algorithm=report['spec'],noise=report['noise_evidence'],
        selected_lambda=report['selected_lambda'],selection_boundary=report['selection_boundary'],
        diagnostics=selected['diagnostics'] if selected else None,seconds=report['seconds']),ensure_ascii=False,indent=2))
    (path/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Signed DCT experimental evidence</title><style>body{max-width:1000px;margin:3em auto;font:17px sans-serif;line-height:1.6}pre{white-space:pre-wrap}</style><h1>Whole Signed DCT experimental results</h1><p>Non-default algorithm. Interpret results using the declared terminal state, scale, noise and basis assumptions. Signed contributions are permitted; peaks are not automatically assigned to H or O. Viewing and exporting do not recompute results.</p><pre>'+body+'</pre>',encoding='utf-8')
    (path/'manifest.json').write_text(json.dumps({p.name:sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    return path
