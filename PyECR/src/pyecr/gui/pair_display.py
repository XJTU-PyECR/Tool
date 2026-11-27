"""Display-only Pair metadata and explicit planning-input serialization."""
import json
from math import isfinite
from numbers import Real
from pyecr.model_display import display_name
from pyecr.diagnostics.interpretation import diagnostic_summary

VERSION='pair-layout/2.0.0'
PAIR_STEPS=('01 Select A & B','02 Check & compare','03 Results & export')
PAIR_KIND_LABELS={'Oxidation / reduction':'oxidation_reduction',
                  'Hydration / dehydration':'hydration_dehydration','Generic':'generic'}
PLAN_STEPS=('01 Ranges and noise','02 Full dimensions and active faces','03 Duration and sampling',
            '04 Boundary assumptions','05 Scan','06 Stored candidates')


def source_caption(member):
    """Display only; the full path and identity remain in the frozen member."""
    if member is None:return 'Capture current analysis or load a completed project.'
    from pathlib import PureWindowsPath
    meta=member.snapshot.dataset.metadata
    source=meta.get('source_path') or meta.get('dataset_name') or member.label
    source=PureWindowsPath(str(source)).name
    return f'{member.label}\n{source} · captured {member.direction}'


def missing_conditions(member):
    if member is None:return ()
    meta={**member.snapshot.dataset.metadata,**member.context}
    return tuple(label for key,label in (('temperature_k','temperature'),('initial_condition','initial condition'),
                                        ('final_condition','final condition')) if meta.get(key) is None)


def short_pair_reason(reason):
    """Keep the policy's full reason in Details; give a concrete next action."""
    text=str(reason);lower=text.lower()
    routes=(('independent datasets','Use two independently analyzed datasets.'),
            ('complementary step','Capture complementary directions for this comparison.'),
            ('direction','Review the captured direction, then capture or load again.'),
            ('geometry','Use matching full dimensions and active faces.'),
            ('unit','Review signal type and units in both source projects.'),
            ('signal type','Review signal type and units in both source projects.'),
            ('do not match','Run current Diagnostics in the source project, then capture or load again.'),
            ('diagnostics','Complete current Fit and Diagnostics, then capture or load again.'),
            ('reliability','Review source Diagnostics; this Fit cannot be compared.'),
            ('identifiability','Review source Diagnostics; parameters are not supported.'),
            ('adequacy','Review source Diagnostics; model adequacy is not supported.'),
            ('temperature','Review captured temperature declarations.'),
            ('condition','Review captured initial and final conditions.'),
            ('models differ','Select sources fitted with the same supported model.'),
            ('boundary','Review boundary assumptions in both source projects.'))
    for needle,action in routes:
        if needle in lower:return action
    return text if len(text)<=155 else text[:152]+'…'


def number(value):
    if isinstance(value,bool):raise ValueError('Table values must be finite numbers, not boolean')
    if not isinstance(value,(str,Real)):raise ValueError('Table values must be finite numbers')
    try:value=float(value)
    except OverflowError as error:raise ValueError('Table value exceeds finite representability') from error
    if not isfinite(value):raise ValueError('Table values must be finite numbers')
    return value


def decode_inputs(values):
    parsed={key:json.loads(text) for key,text in values.items()}
    for key in ('d','k'):
        if not isinstance(parsed[key],list) or len(parsed[key])!=2:raise ValueError(key+' range requires two values')
        parsed[key]=[number(v) for v in parsed[key]]
    if not isinstance(parsed['geometries'],list) or not isinstance(parsed['durations'],list):raise ValueError('Candidate JSON must contain lists')
    if any(not isinstance(row,list) or len(row)!=3 for row in parsed['geometries']):raise ValueError('Each full geometry needs x / y / z')
    parsed['geometries']=[[number(v) for v in row] for row in parsed['geometries']]
    parsed['durations']=[number(v) for v in parsed['durations']]
    return parsed


def encode_draft(draft,original):
    """Validate all table syntax before any widget writes; domain policy stays in Request."""
    parsed=decode_inputs({key:json.dumps(value) for key,value in {
        'd':[number(v) for v in draft['d']], 'k':[number(v) for v in draft['k']],
        'geometries':[[number(v) for v in row] for row in draft['geometries']],
        'durations':[number(v) for v in draft['durations']]}.items()})
    result={}
    for key,value in parsed.items():
        try:unchanged=json.loads(original[key])==value
        except (ValueError,TypeError):unchanged=False
        result[key]=original[key] if unchanged else json.dumps(value,ensure_ascii=False,allow_nan=False)
    return result


def member_text(slot,member,active=None):
    if member is None:return f'{slot}: no captured source. Capture a current Fit + Diagnostics or load its completed project.'
    snapshot=member.snapshot;data=snapshot.dataset;meta=data.metadata;config=snapshot.analysis_config
    relation='Captured snapshot; no active Fit to compare its identity.'
    if active is not None:
        relation=('Current active analysis — captured read-only snapshot' if active.get('matches',False) and not active.get('stale',False)
                  else 'Historical captured snapshot — current Analysis differs or is out of date')
        if active.get('stale',False):relation+='; active Fit / Diagnostics Out of date'
    ops=[record.operation for record in data.history]
    return (f'{slot}: {member.label} | {member.direction}\n{relation}\nModel: {display_name(config.model_id)}; boundary: {config.boundary.mode}\n'
            f'Raw units: {meta.get("raw_time_unit",meta.get("time_unit","unknown"))}, {meta.get("raw_signal_unit",meta.get("signal_unit","unknown"))}\n'
            f'Processed units: {meta.get("time_unit","unknown")}, {meta.get("signal_unit","unknown")}\n'
            f'Preprocessing: {len(ops)} recorded steps; '+(' → '.join(ops) or 'none')+'\n'
            f'Source: {meta.get("source_path",meta.get("dataset_name","captured in-memory data"))}\nFit ID: {member.fit_id}\nDataset SHA256: {snapshot.dataset_sha256}\n'
            f'Full dimensions: {config.model.geometry.full_dims_cm} cm; active pairs: {config.model.geometry.active_dimensions}\n'
            f'Conditions: temperature_k={meta.get("temperature_k","not recorded")}; initial={meta.get("initial_condition","not recorded")}; final={meta.get("final_condition","not recorded")}\n'
            f'Capture declarations: {dict(member.context)}\nSnapshot warnings: '+('; '.join(snapshot.warnings) or 'none')+'\n'+
            diagnostic_summary(snapshot.fit_result,snapshot.diagnostics_result))
