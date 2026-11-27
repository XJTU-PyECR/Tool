"""Readable figure file basenames, separate from immutable scientific IDs."""
import re

FIGURE_NAMES_VERSION='descriptive-files/1'
LEGACY_FIGURE_NAMES_VERSION='figure-id-files/1'
SOURCE_FOLDERS_VERSION='module-folders/1'
LEGACY_SOURCE_FOLDERS_VERSION='source-id-folders/1'
_DIGEST_SUFFIX=re.compile(r'_[0-9a-f]{10}$')
_OPAQUE=re.compile(r'[0-9a-f]{32,64}$')
_DEVICES={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}


def validate_file_stem(stem):
    if not isinstance(stem,str) or not stem or len(stem)>200 or not re.fullmatch(r'[a-zA-Z0-9_.-]+',stem) or stem in ('.','..') or stem.endswith('.') or stem.split('.')[0].upper() in _DEVICES:
        raise ValueError('Unsafe figure filename stem')
    return stem


def figure_file_stems(figures,*,naming_version=FIGURE_NAMES_VERSION):
    """Stable within the complete source catalog, independent of selection.

    Single DCT runs omit their opaque run key. Multiple runs use short stable
    ordinal labels. Readable record/parameter names retain their words while
    dropping safe_id's checksum. Case-insensitive collisions use _02, _03, ... .
    IDs, source hashes, titles, arrays and scientific table names stay intact.
    """
    figures=tuple(figures)
    if len({f.figure_id for f in figures})!=len(figures):raise ValueError('Duplicate figure identity')
    if naming_version==LEGACY_FIGURE_NAMES_VERSION:return {f.figure_id:validate_file_stem(f.figure_id) for f in figures}
    if naming_version!=FIGURE_NAMES_VERSION:raise ValueError('Unknown figure filename naming version')
    runs=[];contexts=[]
    for f in figures:
        p=f.figure_id.split('.')
        if len(p)>=3 and p[0]=='dct' and p[1] not in ('monotonic','context','unavailable') and p[1] not in runs:runs.append(p[1])
        if len(p)>=5 and p[:2]==['dct','context'] and tuple(p[2:-1]) not in contexts:contexts.append(tuple(p[2:-1]))
    bases=[]
    for f in figures:
        p=f.figure_id.split('.')
        if len(p)>=3 and p[0]=='dct' and p[1] in runs:
            p=['dct',*([f'run_{runs.index(p[1])+1:02d}'] if len(runs)>1 else []),*p[2:]]
        elif len(p)>=5 and p[:2]==['dct','context']:
            key=tuple(p[2:-1]);p=['dct','context',*([f'record_{contexts.index(key)+1:02d}'] if len(contexts)>1 else []),p[-1]]
        clean=[]
        for part in p:
            part=_DIGEST_SUFFIX.sub('',part)
            if part and not _OPAQUE.fullmatch(part):clean.append(part)
        stem='.'.join(clean).strip('.')[:140].rstrip('.') or 'figure'
        if stem.split('.')[0].upper() in _DEVICES:stem='figure.'+stem
        bases.append(validate_file_stem(stem))
    used=set();reserved={b.casefold() for b in bases};result={}
    for f,base in zip(figures,bases,strict=True):
        name=base;i=2
        # Reserve later natural names as well as already emitted filenames.
        while name.casefold() in used or (name!=base and name.casefold() in reserved):
            name=f'{base}_{i:02d}';i+=1
        used.add(name.casefold());result[f.figure_id]=validate_file_stem(name)
    return result


def unique_stems(stems,*,max_length=27):
    """Short case-insensitive names; reserve natural suffixes before allocating.

    The table limit reserves four Excel characters for continuation sheets.
    All names are allocated from the complete catalog before figure selection.
    """
    bases=[validate_file_stem(b[:max_length].rstrip('.')) for b in stems]
    used=set();reserved={b.casefold() for b in bases};result=[]
    for base in bases:
        name=base;i=2
        while name.casefold() in used or (name!=base and name.casefold() in reserved):
            suffix=f'_{i:02d}';name=base[:max_length-len(suffix)].rstrip('._')+suffix;i+=1
        used.add(name.casefold());result.append(validate_file_stem(name))
    return tuple(result)


def source_folder_names(identities,*,naming_version=SOURCE_FOLDERS_VERSION):
    """Map (immutable source ID, module) pairs to readable output folders."""
    identities=tuple(identities)
    if len({sid for sid,_ in identities})!=len(identities):raise ValueError('Duplicate source identity')
    if naming_version==LEGACY_SOURCE_FOLDERS_VERSION:
        return {sid:validate_file_stem(sid) for sid,_ in identities}
    if naming_version!=SOURCE_FOLDERS_VERSION:raise ValueError('Unknown source folder naming version')
    modules=[module.lower() for _,module in identities]
    if any(m not in ('analysis','batch','global','pair','dct','simulation') for m in modules):raise ValueError('Unknown source folder module')
    counts={m:modules.count(m) for m in modules};seen={};result={}
    for (sid,_),module in zip(identities,modules,strict=True):
        validate_file_stem(sid);seen[module]=seen.get(module,0)+1
        result[sid]=module if counts[module]==1 else f'{module}_{seen[module]:02d}'
    return result
