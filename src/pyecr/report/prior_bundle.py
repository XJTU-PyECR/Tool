"""Read-only export of a frozen experimental Signed archive into a NEW folder."""
from pathlib import Path,PurePosixPath
from hashlib import sha256
import csv,json,html
from pyecr.contracts.common import require,canonical_sha256,strict_json_loads
from pyecr.project.prior_archive import SignedPriorArchive
from pyecr.gui.prior_evidence import prior_display,prior_scan


def filehash(p):return sha256(p.read_bytes()).hexdigest()
def wire(value):return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)


def export_prior_bundle(archive,directory):
    archive=archive.validated();root=Path(directory).resolve();require(not root.exists(),'experimental export requires a new directory')
    root.mkdir(parents=True);view=prior_display(archive,physical=False);physical=prior_display(archive,physical=True);scan=prior_scan(archive)
    (root/'prior_archive.json').write_text(wire(archive.to_dict()),encoding='utf-8')
    (root/'presentation.json').write_text(wire(dict(normalized={n:getattr(view,n) for n in view.__dataclass_fields__},
        physical={n:getattr(physical,n) for n in physical.__dataclass_fields__},common_scale=archive.report.source.normalization.scale_value,independent_shape=False)),encoding='utf-8')
    def table(name,header,rows):
        with (root/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(header)
            for row in rows:w.writerow([format(float(v),'.17g') if isinstance(v,(float,int)) and not isinstance(v,bool) else v for v in row])
    table('candidates.csv',('lambda','score','status','failure','seconds','subproblems'),
        [(t.result.config.grid.regularization_lambda,t.score,t.result.prior_evidence.status,t.failure,t.seconds,t.subproblems) for t in archive.report.trials])
    selected=archive.report.selected_result
    if selected:
        table('spectrum.csv',('tau_s','log10_tau','chi','q','m','physical_chi'),zip(selected.tau_s,selected.log10_tau,selected.chi,selected.quadrature_weights,selected.quadrature_mass,selected.physical_chi))
        table('reconstruction.csv',('time_s','observed_Y','predicted_Y','residual_Y','observed_physical','predicted_physical','residual_physical'),
            zip(view.time,view.observed,view.reconstruction,view.residual,physical.observed,physical.reconstruction,physical.residual))
    embedded=wire(dict(selected=scan.selected,time=scan.time,observed=scan.observed,score_name=scan.score_name,
        candidates=[dict(lambda_value=c.lam,score=c.score,tau=c.tau,chi=c.chi,prediction=c.reconstruction,note=c.note) for c in scan.candidates])).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>Experimental Signed evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{width:100%;height:320px;border:1px solid #ddd}select{font:inherit}path{fill:none;stroke:#0072b2;stroke-width:1.5}</style>
<h1>Experimental Signed evidence has not passed validation</h1><p>Frozen stored spectra. Browsing never solves, changes lambda, publishes current, fits a physical model or deconvolves.</p>
<p>'''+html.escape(view.summary)+'''</p><select id="choice"></select><p id="note"></p><svg id="spectrum" viewBox="0 0 1000 300"></svg>
<h2>Frozen reconstruction</h2><svg id="reconstruction" viewBox="0 0 1000 300"></svg><pre>'''+html.escape(view.details)+'''</pre>
<script type="application/json" id="data">'''+embedded+'''</script><script>
const data=JSON.parse(document.getElementById('data').textContent),choice=document.getElementById('choice');
data.candidates.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent='lambda='+c.lambda_value+(c.lambda_value===data.selected?' [selected]':'');choice.appendChild(o);if(c.lambda_value===data.selected)choice.value=i;});
function plot(id,x,y){let s=document.getElementById(id);s.replaceChildren();if(!x.length||x.length!==y.length)return;let xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(0,...y),ymax=Math.max(0,...y);if(ymin===ymax)ymax=ymin+1;let d=x.map((v,i)=>(i?'L':'M')+(30+940*(v-xmin)/(xmax-xmin||1))+','+(270-240*(y[i]-ymin)/(ymax-ymin))).join(' ');let p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',d);s.appendChild(p);}
function view(){let c=data.candidates[+choice.value];if(!c)return;document.getElementById('note').textContent=data.score_name+': '+c.score+' | '+c.note+' | normalized common S, independent_shape=False';plot('spectrum',c.tau.map(Math.log10),c.chi);plot('reconstruction',data.time,c.prediction);}
choice.onchange=view;view();</script>'''
    (root/'report.html').write_text(page,encoding='utf-8')
    manifest=dict(kind='signed_prior_bundle',schema_version=1,archive_sha256=canonical_sha256(archive.to_dict()),files={p.name:filehash(p) for p in root.iterdir() if p.is_file()})
    (root/'manifest.json').write_text(wire(manifest),encoding='utf-8');read_prior_bundle(root);return root


def read_prior_bundle(directory):
    root=Path(directory).resolve();m=strict_json_loads((root/'manifest.json').read_text(encoding='utf-8'))
    require(set(m)=={'kind','schema_version','archive_sha256','files'} and m['kind']=='signed_prior_bundle' and type(m['schema_version']) is int and m['schema_version']==1,'unsupported prior report bundle')
    require({p.name for p in root.iterdir()}==set(m['files'])|{'manifest.json'},'prior bundle file inventory conflict')
    for name,h in m['files'].items():
        p=PurePosixPath(name);require(len(p.parts)==1 and not p.is_absolute() and ':' not in name and '\\' not in name and name not in {'.','..'},'unsafe prior artifact path')
        file=root/name;require(not file.is_symlink() and file.is_file() and filehash(file)==h,'prior artifact hash conflict')
    obj=SignedPriorArchive.from_json((root/'prior_archive.json').read_text(encoding='utf-8'));require(isinstance(obj,SignedPriorArchive),'unsupported prior evidence')
    require(canonical_sha256(obj.to_dict())==m['archive_sha256'],'prior archive content conflict')
    saved=strict_json_loads((root/'presentation.json').read_text(encoding='utf-8'))
    for key,physical in (('normalized',False),('physical',True)):
        view=prior_display(obj,physical=physical);require(canonical_sha256(saved[key])==canonical_sha256({n:getattr(view,n) for n in view.__dataclass_fields__}),'GUI/report frozen presentation conflict')
    require(saved['independent_shape'] is False and saved['common_scale']==obj.report.source.normalization.scale_value,'common S/shape identity conflict')
    if obj.report.selected_result:
        with (root/'spectrum.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        r=obj.report.selected_result;require(len(rows)==len(r.tau_s),'spectrum row loss')
        for row,values in zip(rows,zip(r.tau_s,r.log10_tau,r.chi,r.quadrature_weights,r.quadrature_mass,r.physical_chi)):
            require(tuple(float(row[k]) for k in ('tau_s','log10_tau','chi','q','m','physical_chi'))==values,'stored scientific spectrum conflict')
        with (root/'reconstruction.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        normalized=prior_display(obj);physical=prior_display(obj,physical=True)
        require(len(rows)==len(r.source.signal),'observation row loss')
        expected=zip(normalized.time,normalized.observed,normalized.reconstruction,normalized.residual,physical.observed,physical.reconstruction,physical.residual)
        for row,values in zip(rows,expected):
            require(tuple(float(row[k]) for k in ('time_s','observed_Y','predicted_Y','residual_Y','observed_physical','predicted_physical','residual_physical'))==values,'stored scientific reconstruction conflict')
    return obj
