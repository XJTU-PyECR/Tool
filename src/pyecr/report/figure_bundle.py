"""Selected multi-source figures, full precision arrays and atomic new output."""
from dataclasses import dataclass,field
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from html import escape
import csv,json,os,shutil
import re
import numpy as np
from pyecr.report.figure_catalog import VERSION,FigureSource,catalog,freeze,thaw,digest
from pyecr.report.export_options import ExportOptions
from pyecr.report.export_tables import ExportTable
from pyecr.report.export_v2 import _write_table
from pyecr.report.export_readback import file_sha256,_decode
from pyecr.report.series_tables import (TABLE_LAYOUT_VERSION,
    LEGACY_TABLE_LAYOUT_VERSION,build_series_table)
from pyecr.report.figure_names import (FIGURE_NAMES_VERSION,LEGACY_FIGURE_NAMES_VERSION,figure_file_stems,
    SOURCE_FOLDERS_VERSION,LEGACY_SOURCE_FOLDERS_VERSION,source_folder_names,unique_stems)

@dataclass(frozen=True)
class FigureExportOptions:
    figure_ids: tuple | None=None  # source_id/figure_id, or None = All available
    formats: tuple=('png',)
    include_xlsx: bool=False
    style: ExportOptions=field(default_factory=ExportOptions)
    allow_arrays_only: bool=False
    def __post_init__(self):
        object.__setattr__(self,'formats',tuple(self.formats))
        if self.figure_ids is not None:
            object.__setattr__(self,'figure_ids',tuple(self.figure_ids))
            if len(set(self.figure_ids))!=len(self.figure_ids):raise ValueError('Duplicate figure selection')
        if not self.formats or set(self.formats)-{'png','svg','pdf'} or len(set(self.formats))!=len(self.formats):raise ValueError('Select PNG, SVG or figure PDF formats')
        if type(self.include_xlsx) is not bool:raise TypeError('include_xlsx must be boolean')
        if type(self.allow_arrays_only) is not bool:raise TypeError('allow_arrays_only must be boolean')
        if not isinstance(self.style,ExportOptions):raise TypeError('Figure ExportOptions required')
    def to_dict(self):
        value=dict(version=VERSION,figure_ids=self.figure_ids,formats=self.formats,include_xlsx=self.include_xlsx,style=self.style.to_dict())
        if self.allow_arrays_only:value['allow_arrays_only']=True
        return value
    @classmethod
    def from_dict(cls,value):
        if value.get('version')!=VERSION:raise ValueError('Unknown figure selection version')
        return cls(value['figure_ids'],value['formats'],value['include_xlsx'],ExportOptions(**value['style']),value.get('allow_arrays_only',False))

@dataclass(frozen=True)
class FigureBundle:
    root: Path
    manifest: dict
    def verify(self):return read_figure_bundle(self.root)

TABLE_NAMES_VERSION='descriptive/2'
HASHED_TABLE_NAMES_VERSION='descriptive/1'
LEGACY_TABLE_NAMES_VERSION='index-hash/1'


def _figure_table_name(figure,index,naming_version):
    if naming_version==LEGACY_TABLE_NAMES_VERSION:
        return 'F'+str(index+1).zfill(4)+'_'+digest(figure.figure_id)[:10]
    if naming_version not in (TABLE_NAMES_VERSION,HASHED_TABLE_NAMES_VERSION):
        raise ValueError('Unknown figure table naming version')
    # Put the data's role first: truncating a long title used to hide the only
    # word distinguishing reconstruction/residuals or clean/observed tables.
    # The full, original title remains in the HTML index and figures.json.
    parts=figure.figure_id.split('.')
    role=parts[-1]
    roles=dict(raw='RawData',processed='ProcessedData',fit='Fit',curve='Fit',
        curves='ABFits',residuals='Residuals',components='Components',
        components_early='EarlyComponents',components_residuals='ComponentResiduals',
        sensitivity='Sensitivity',sensitivity_normalized='ScaledSensitivity',
        transport='TransportBi',spectrum='Spectrum',reconstruction='Reconstruction',
        weighted_residuals='WeightedResiduals',global_covariance='Covariance',
        global_correlation='Correlation',overlay='CleanObserved',contributions='Contributions',
        clean='Clean',observed='Observed',noise='Noise',given_estimated='GivenEstimated',
        early_mismatch='EarlyMismatch',candidates='DesignCandidates',
        **{'lambda':'LambdaGCV'},stability='PeakStability',peak_stability='PeakStability',
        physical_dct='PhysicalResponse',physical_spectrum='PhysicalSpectrum',
        physical_residuals='PhysicalResiduals',independent_shapes='ComponentShapes',
        common_scale='CommonScale',component_time='ComponentCurves',
        component_residuals='ComponentResiduals',shape='Shape',modes='AppliedModes',
        forward='AppliedForward')
    prefix=figure.module
    label=''
    if parts[0]=='error_map':
        prefix='Analysis';role='ErrorMap_'+role
    elif parts[0] in ('analysis','pair') and parts[1] in ('interval','bootstrap','profile','parameter'):
        # Remove only the catalog's identity suffix, never a scientific unit.
        label=re.sub(r'_[a-f0-9]{10}$','',role)
        role=dict(interval='Interval',bootstrap='Bootstrap',profile='Profile',parameter='Parameter')[parts[1]]
    elif parts[0]=='batch' and parts[1] in ('parameters','arrhenius'):
        role='Parameters' if parts[1]=='parameters' else 'Arrhenius'
        label=re.sub(r'_[a-f0-9]{10}$','',parts[-1])
        if parts[1]=='arrhenius':label=re.sub(r'^M\d+_[A-Z_]+_','',label)
    elif parts[0]=='global' and parts[1]=='shared':
        role='Shared';label=re.sub(r'_[a-f0-9]{10}$','',parts[-1])
    elif parts[0] in ('batch','global') and role in ('curve','residuals','weighted_residuals'):
        label=re.sub(r'_[a-f0-9]{10}$','',parts[1])
    role=roles.get(role,role)
    if parts[0]=='dct':
        if parts[1]=='monotonic':prefix='DCT_Mono'
        if len(parts)>3 and parts[-2].startswith('component_'):
            prefix='DCT_C'+parts[-2].removeprefix('component_')
    title='_'.join(v for v in (prefix,role,label) if v)
    # New names reserve four Excel characters for continuation sheets. Keep
    # the previous truncation/hash rule exact for historical package readback.
    if naming_version==TABLE_NAMES_VERSION:
        return re.sub('[^a-zA-Z0-9]+','_',title).strip('_')[:27].rstrip('_')
    stem=re.sub('[^a-zA-Z0-9]+','_',title).strip('_')[:20].rstrip('_')
    return stem+'_'+digest(figure.figure_id)[:6]


def figure_tables(figures,*,naming_version=TABLE_NAMES_VERSION,layout_version=TABLE_LAYOUT_VERSION):
    if layout_version not in (TABLE_LAYOUT_VERSION,LEGACY_TABLE_LAYOUT_VERSION):
        raise ValueError('Unknown figure table layout version')
    if naming_version not in (TABLE_NAMES_VERSION,HASHED_TABLE_NAMES_VERSION,LEGACY_TABLE_NAMES_VERSION):
        raise ValueError('Unknown figure table naming version')
    figures=tuple(figures)
    raw_names=[_figure_table_name(f,i,naming_version) for i,f in enumerate(figures) if f.available]
    allocated=iter(unique_stems(raw_names) if naming_version==TABLE_NAMES_VERSION else raw_names)
    tables=[]
    names=set()
    for i,f in enumerate(figures):
        if not f.available:continue
        if f.renderer=='objective_map':
            p=thaw(f.payload);r=p['result'];grid=r['objective_grid' if p['process']=='single' else 'raw_sse_grid']
            rows=tuple((a,b,float(10**a),float(10**b),grid[x][y],bool(r['valid_mask'][x][y])) for x,a in enumerate(r['log10_D']) for y,b in enumerate(r['log10_k']))
            columns=('log10_D','log10_k','D_cm2_s','k_cm_s','raw_SSE','valid');types=('float64',)*4+('float64?','boolean')
            units=dict(log10_D='decade(cm²/s)',log10_k='decade(cm/s)',D_cm2_s='cm²/s',k_cm_s='cm/s',raw_SSE=p['signal_unit']+'²')
        elif layout_version==LEGACY_TABLE_LAYOUT_VERSION:
            rows=tuple((s.label,s.series_id,s.sampling,j,None if s.kind=='hline' else s.x[j],v) for s in f.series for j,v in enumerate(s.y))
            columns=('series','series_id','sampling','index','x','y');types=('text','text','text','integer','float64?','float64')
            units={'x':f.x_unit,'y':f.y_unit}
        name=next(allocated)
        if name.lower() in names:raise ValueError('Figure table name collision')
        names.add(name.lower())
        table=(build_series_table(f,name) if f.renderer!='objective_map' and layout_version==TABLE_LAYOUT_VERSION
               else ExportTable(name,columns,types,units,len(rows),lambda rows=rows:iter(rows)))
        tables.append((f.figure_id,table))
    return tuple(tables)

def _json(path,value):path.write_text(freeze(value)+'\n',encoding='utf-8',newline='\n')

def _verify_format(path,fmt):
    if path.stat().st_size<100:raise ValueError('Empty figure file: '+path.name)
    if fmt=='png':
        from PIL import Image
        with Image.open(path) as im:
            if im.format!='PNG' or min(im.size)<10:raise ValueError('Invalid PNG figure')
            im.verify()
    elif fmt=='svg':
        import xml.etree.ElementTree as ET
        root=ET.parse(path).getroot()
        if not root.tag.endswith('svg') or not any(v.tag.endswith(('path','image','text')) for v in root.iter()):raise ValueError('Invalid SVG figure')
    elif fmt=='pdf':
        # The application has no optional full-document PDF reader dependency.
        # Matplotlib writes vector figure PDFs; native/full PDF opening is a
        # separate acceptance step, never a configured full-text renderer.
        data=path.read_bytes()
        if not data.startswith(b'%PDF-') or b'%%EOF' not in data[-128:] or b'/Type /Page' not in data:raise ValueError('Invalid figure PDF envelope')

def export_figure_bundle(sources,destination,*,options=None,cancel_check=None,before_publish=None,progress_callback=None):
    sources=tuple(sources);options=options or FigureExportOptions()
    if not sources or any(not isinstance(s,FigureSource) for s in sources):raise ValueError('Select at least one captured result source')
    if len({s.source_id for s in sources})!=len(sources):raise ValueError('Select each source identity once')
    if not isinstance(options,FigureExportOptions):raise TypeError('FigureExportOptions required')
    target=Path(destination).expanduser()
    if not target.is_absolute():raise ValueError('Enter an absolute new export directory')
    if target.is_symlink():raise ValueError('Export destination must not be a link')
    target=target.resolve()
    if target==target.parent:raise ValueError('Export destination must name a new directory')
    if target.exists():raise FileExistsError('Figure export destination exists; choose a new directory')
    target.parent.mkdir(parents=True,exist_ok=True)
    lock=target.parent/('.'+target.name+'.figures.lock');handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(handle)
    staging=None
    def check():
        if cancel_check:cancel_check()
    def progress(v,msg):
        if progress_callback:progress_callback(v,msg)
    try:
        check()
        catalogs={s.source_id:catalog(s) for s in sources}
        folders=source_folder_names(((s.source_id,s.module) for s in sources),naming_version=SOURCE_FOLDERS_VERSION)
        available={s.source_id+'/'+f.figure_id for s in sources for f in catalogs[s.source_id] if f.available}
        selected=available if options.figure_ids is None else set(options.figure_ids)
        if not selected and not options.allow_arrays_only:raise ValueError('Select at least one available figure')
        if selected-available:raise ValueError('Selected figure is unavailable or belongs to another source')
        from pyecr.report.staging import create_export_staging
        staging=create_export_staging(target.parent,prefix='.'+target.name+'.figures-tmp-');records=[];source_records=[];table_links={}
        for source in sources:
            check();folder=staging/folders[source.source_id];folder.mkdir();(folder/'figures').mkdir()
            _json(folder/'source.json',source.to_dict());figures=catalogs[source.source_id]
            _json(folder/'catalog.json',dict(version=VERSION,figures=[f.to_dict() for f in figures]))
            file_stems=figure_file_stems(figures,naming_version=FIGURE_NAMES_VERSION)
            tables=figure_tables(figures);mapping={fid:t for fid,t in tables}
            table_links[source.source_id]=tuple((f.title,mapping[f.figure_id].info()['file']) for f in figures if f.figure_id in mapping)
            for fid,table in tables:check();_write_table(folder,table,options.style,check)
            xlsx=None
            if options.include_xlsx:
                from pyecr.report.export_xlsx import write_xlsx
                xlsx=write_xlsx(tuple(t for _,t in tables),folder/'arrays.xlsx',options.style,
                    dict(source_id=source.source_id,precision='XLSX 15 digits; exact source.json and 17g CSV'),check)
            source_records.append(dict(source_id=source.source_id,module=source.module,result_type=source.result_type,result_sha256=source.result_sha256,
                scope=source.scope,label=source.label,tables={fid:t.info() for fid,t in tables},xlsx=xlsx))
            if SOURCE_FOLDERS_VERSION!=LEGACY_SOURCE_FOLDERS_VERSION:source_records[-1]['folder']=folders[source.source_id]
            for f in figures:
                check();key=source.source_id+'/'+f.figure_id;chosen=key in selected;files=[];rendered=None
                if chosen:
                    from pyecr.report.figure_render import write_figure
                    rendered=write_figure(f,folder/'figures',options.formats,options.style,check,filename_stem=file_stems[f.figure_id])
                    for fmt in options.formats:
                        file=folder/'figures'/(file_stems[f.figure_id]+'.'+fmt);_verify_format(file,fmt)
                        files.append(dict(file=file.relative_to(staging).as_posix(),format=fmt,bytes=file.stat().st_size,sha256=file_sha256(file)))
                records.append(dict(figure_id=f.figure_id,source_id=source.source_id,module=f.module,result_type=f.result_type,title=f.title,plot_id=f.plot_id,
                    available=f.available,unavailable_reason=f.unavailable_reason,selected=chosen,files=files,rendered=rendered,
                    table=None if f.figure_id not in mapping else mapping[f.figure_id].info()['file']))
                progress(.1+.65*sum(r['selected'] for r in records)/max(1,len(selected)),'Exported '+f.title if chosen else 'Indexed '+f.title)
        _json(staging/'figures.json',dict(version=VERSION,table_naming_version=TABLE_NAMES_VERSION,
            table_layout_version=TABLE_LAYOUT_VERSION,figure_naming_version=FIGURE_NAMES_VERSION,source_folder_version=SOURCE_FOLDERS_VERSION,sources=source_records,figures=records))
        html=['<!doctype html><html><head><meta charset="utf-8"><title>PyECR figures</title></head><body><h1>Captured result figures</h1>',
            '<p>Scientific arrays are complete in each source JSON and CSV. Figure PDFs are graphics, not a full-text report PDF. XLSX numeric precision is 15 digits.</p>',
            '<p>Each curve has its own value column. Identical coordinates share one column; different grids use separate coordinate/value pairs. Empty cells mean no sample, not zero. Reference constants are stored once. Column identities, units and sampling are in figures.json.</p>']
        for s in sources:
            html.append('<h2>'+escape(s.label or s.module)+' — '+escape(s.scope)+'</h2><p>Source '+s.source_id+'; SHA256 '+s.result_sha256+'</p>')
            from .capability_evidence import source_records,caption
            for evidence in source_records(s):
                e=evidence['eligibility']
                html.append('<p>'+escape(caption(e))+f' Policy: {escape(e["policy_version"])}; r1={e["residual_lag1"]}; abs limit={e["residual_limit"]}.</p>')
            for r in (r for r in records if r['source_id']==s.source_id):
                if r['selected']:
                    html.append('<h3>'+escape(r['title'])+'</h3>')
                    for f in r['files']:
                        path=escape(f['file'],quote=True)
                        html.append('<img style="max-width:100%" src="'+path+'" alt="'+escape(r['title'],quote=True)+'">' if f['format']=='png' else '<a href="'+path+'">'+f['format'].upper()+'</a>')
                elif not r['available']:html.append('<p>'+escape(r['title'])+': '+escape(r['unavailable_reason'])+'</p>')
            html.append('<h3>Scientific tables</h3><ul>')
            for title,file in table_links[s.source_id]:
                html.append('<li><a href="'+escape(folders[s.source_id]+'/'+file,quote=True)+'">'+escape(title)+' — full data CSV</a></li>')
            html.append('</ul>')
        html.append('</body></html>');(staging/'report.html').write_text('\n'.join(html)+'\n',encoding='utf-8')
        artifacts={p.relative_to(staging).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest=dict(kind='pyecr_figure_bundle',version=VERSION,exported_at=datetime.now(timezone.utc).isoformat(),options=options.to_dict(),sources=[s.source_id for s in sources],artifacts=artifacts)
        evidence={s.source_id:source_records(s) for s in sources if source_records(s)}
        if evidence:manifest['capability_evidence']=evidence
        manifest['manifest_sha256']=digest(manifest);_json(staging/'manifest.json',manifest)
        progress(.85,'Checking hashes, complete arrays, units, selection and figure files')
        read_figure_bundle(staging,check=check);check()
        if before_publish:before_publish()
        check()
        if target.exists():raise FileExistsError('Destination appeared during export; previous output retained')
        # New-only publication, as in the independent Simulation exporter.
        # Windows rename atomically refuses an existing target, including one
        # created after the check above. The project replacement helper would
        # intentionally back up/delete that conflicting directory.
        staging.rename(target);staging=None
        return FigureBundle(target,manifest)
    finally:
        if staging is not None and staging.exists():
            if staging.resolve().parent!=target.parent or not staging.name.startswith('.'+target.name+'.figures-tmp-'):raise RuntimeError('Refusing cleanup outside owned figure staging')
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)

def read_figure_bundle(root,*,check=lambda:None):
    root=Path(root)
    if root.is_symlink():raise ValueError('Figure bundle must not be a link')
    root=root.resolve();manifest=thaw((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='pyecr_figure_bundle' or manifest.get('version')!=VERSION:raise ValueError('Unknown figure bundle version')
    if manifest['manifest_sha256']!=digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):raise ValueError('Figure manifest hash conflict')
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(manifest['artifacts'])|{'manifest.json'}:raise ValueError('Figure artifact inventory conflict')
    for name,info in manifest['artifacts'].items():
        check();p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe figure artifact path')
        file=root/name
        if not file.resolve().is_relative_to(root) or any(v.is_symlink() for v in (file,*file.parents) if v!=root.parent):raise ValueError('Figure artifact link/outside root')
        if file.stat().st_size!=info['bytes'] or file_sha256(file)!=info['sha256']:raise ValueError('Figure artifact hash conflict: '+name)
    options=FigureExportOptions.from_dict(manifest['options']);index=thaw((root/'figures.json').read_text(encoding='utf-8'))
    if index['version']!=VERSION or [s['source_id'] for s in index['sources']]!=manifest['sources']:raise ValueError('Figure source index conflict')
    folder_naming=index.get('source_folder_version',LEGACY_SOURCE_FOLDERS_VERSION)
    folders=source_folder_names(((s['source_id'],s['module']) for s in index['sources']),naming_version=folder_naming)
    for info in index['sources']:
        if (folder_naming!=LEGACY_SOURCE_FOLDERS_VERSION and info.get('folder')!=folders[info['source_id']]) or ('folder' in info and info['folder']!=folders[info['source_id']]):raise ValueError('Source folder identity conflict')
    figure_naming=index.get('figure_naming_version',LEGACY_FIGURE_NAMES_VERSION)
    if figure_naming not in (FIGURE_NAMES_VERSION,LEGACY_FIGURE_NAMES_VERSION):raise ValueError('Unknown figure filename naming version')
    naming_version=index.get('table_naming_version',LEGACY_TABLE_NAMES_VERSION)
    if naming_version not in (TABLE_NAMES_VERSION,HASHED_TABLE_NAMES_VERSION,LEGACY_TABLE_NAMES_VERSION):raise ValueError('Unknown figure table naming version')
    layout_version=index.get('table_layout_version',LEGACY_TABLE_LAYOUT_VERSION)
    if layout_version not in (TABLE_LAYOUT_VERSION,LEGACY_TABLE_LAYOUT_VERSION):raise ValueError('Unknown figure table layout version')
    seen=[];capabilities={}
    for source_info in index['sources']:
        folder=root/folders[source_info['source_id']];source=FigureSource.from_dict(thaw((folder/'source.json').read_text(encoding='utf-8')))
        from .capability_evidence import source_records
        evidence=source_records(source)
        if evidence:capabilities[source.source_id]=evidence
        if any(source_info[k]!=getattr(source,k) for k in ('source_id','module','result_type','result_sha256','scope','label')):raise ValueError('Figure source metadata binding conflict')
        figures=catalog(source);stored=thaw((folder/'catalog.json').read_text(encoding='utf-8'))
        file_stems=figure_file_stems(figures,naming_version=figure_naming)
        if freeze(stored)!=freeze(dict(version=VERSION,figures=[f.to_dict() for f in figures])):raise ValueError('Figure catalog/full arrays conflict')
        tables=figure_tables(figures,naming_version=naming_version,layout_version=layout_version)
        if source_info['tables']!={fid:t.info() for fid,t in tables}:raise ValueError('Figure table units/count conflict')
        for _,table in tables:
            with (folder/table.info()['file']).open(encoding=options.style.csv_encoding,newline='') as stream:
                reader=csv.reader(stream,delimiter=options.style.csv_delimiter)
                if next(reader,None)!=list(table.columns):raise ValueError('Figure CSV header conflict')
                for actual,expected in zip(reader,table.rows(),strict=True):
                    check()
                    decoded=tuple(_decode(v,k) for v,k in zip(actual,table.types,strict=True))
                    if freeze(decoded)!=freeze(expected):raise ValueError('Figure CSV full precision conflict: '+table.name)
        if source_info['xlsx'] is not None:
            from pyecr.report.export_xlsx import verify_xlsx_tables
            verify_xlsx_tables(tuple(t for _,t in tables),folder/'arrays.xlsx',source_info['xlsx'],check)
        rows=[r for r in index['figures'] if r['source_id']==source.source_id]
        if [r['figure_id'] for r in rows]!=[f.figure_id for f in figures]:raise ValueError('Figure coverage/ordering conflict')
        for f,r in zip(figures,rows,strict=True):
            key=source.source_id+'/'+f.figure_id;seen.append(key)
            expected=f.available and (options.figure_ids is None or key in options.figure_ids)
            if r['selected']!=expected or r['available']!=f.available or r['unavailable_reason']!=f.unavailable_reason:raise ValueError('Figure selection/availability conflict')
            if any(r[k]!=getattr(f,k) for k in ('title','module','result_type','plot_id')):raise ValueError('Figure descriptor metadata conflict')
            if expected and (r['rendered']['x_unit']!=f.x_unit or r['rendered']['y_unit']!=f.y_unit):raise ValueError('Figure rendered units conflict')
            if [v['format'] for v in r['files']]!=(list(options.formats) if expected else []):raise ValueError('Figure format coverage conflict')
            for file in r['files']:
                if file['file']!=folders[source.source_id]+'/figures/'+file_stems[f.figure_id]+'.'+file['format'] or file['sha256']!=file_sha256(root/file['file']):raise ValueError('Figure file/source hash conflict')
                _verify_format(root/file['file'],file['format'])
    if options.figure_ids is not None and set(options.figure_ids)-set(seen):raise ValueError('Unknown selected figure')
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.files=[]
        def handle_starttag(self,tag,attrs):
            if tag in ('img','a'):
                data=dict(attrs);self.files.append(data.get('src' if tag=='img' else 'href',''))
    links=Links();links.feed((root/'report.html').read_text(encoding='utf-8'))
    expected_links=[f['file'] for r in index['figures'] for f in r['files']]
    if naming_version in (TABLE_NAMES_VERSION,HASHED_TABLE_NAMES_VERSION):
        expected_links.extend(folders[s['source_id']]+'/'+t['file'] for s in index['sources'] for t in s['tables'].values())
    if sorted(links.files)!=sorted(expected_links):raise ValueError('HTML figure/table file coverage conflict')
    if manifest.get('capability_evidence',{})!=capabilities:raise ValueError('Figure manifest eligibility conflict')
    return FigureBundle(root,manifest)
