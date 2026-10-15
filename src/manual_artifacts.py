"""Portable shared/seed ZIPs, with identity checks before merging local results."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile
from io_utils import read, file_hash

MANIFEST = 'NAFC_BUNDLE.json'
REPORT_FILES = ('comparison.json', 'REPORT.md', 'overview.png', 'trajectory.png')


def pack(output, archive, seed=None):
    """Pack shared preparation, one seed, or the full notebook (seed='all')."""
    output, archive = Path(output).resolve(), Path(archive).resolve()
    identity = read(output/'shared'/'identity.json')
    folders = [output/'shared'] if seed in (None, 'all') else [output/f'seed_{seed}']
    if seed == 'all':
        folders += [output/f'seed_{value}' for value in identity['config']['seeds'] if (output/f'seed_{value}').exists()]
    if archive.is_relative_to(output) or any(not folder.is_dir() for folder in folders):
        raise ValueError('Save archives outside outputs; the selected results folder must exist.')
    prepared = output/'shared'/'prepared.json'
    manifest = {'format': 1, 'identity': identity, 'seed': seed,
                'prepared_sha256': file_hash(prepared) if prepared.exists() else None}
    if ((seed not in (None, 'all') or len(folders) > 1) and manifest['prepared_sha256'] is None):
        raise ValueError('Seed results need their frozen shared preparation.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(archive)+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            bundle.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False))
            paths = [path for folder in folders for path in folder.rglob('*')]
            if seed in (None, 'all'): paths += [output/name for name in REPORT_FILES if (output/name).exists()]
            for path in sorted(paths):
                if path.is_symlink():
                    raise ValueError('Result archives cannot contain symbolic links.')
                if path.is_file() and not path.name.endswith('.tmp'):
                    bundle.write(path, path.relative_to(output).as_posix())
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def unpack(archives, output, config):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    files, identity, seeds = {}, None, []
    shared_identity = output/'shared'/'identity.json'
    if shared_identity.exists(): identity = read(shared_identity)
    for archive in map(Path, archives):
        with zipfile.ZipFile(archive) as bundle:
            if len(bundle.namelist()) != len(set(bundle.namelist())):
                raise ValueError('Duplicate archive members.')
            meta = json.loads(bundle.read(MANIFEST))
            if meta.get('format') != 1 or meta['identity']['config'] != config:
                raise ValueError('Archive model/configuration differs from this notebook.')
            if identity is not None and identity != meta['identity']:
                raise ValueError('Do not mix different models, environments or shared preparations.')
            identity = meta['identity']
            seed = meta['seed']
            if seed not in (None, 'all') and seed not in config['seeds']:
                raise ValueError('Archive contains an undeclared seed.')
            prefixes = {'shared'} if seed in (None, 'all') else {f'seed_{seed}'}
            if seed == 'all': prefixes |= {f'seed_{value}' for value in config['seeds']}
            if seed not in (None, 'all'): seeds.append(meta['prepared_sha256'])
            for info in bundle.infolist():
                if info.filename == MANIFEST: continue
                path = PurePosixPath(info.filename)
                if (info.orig_filename != info.filename or '\\' in info.filename or ':' in info.filename or path.is_absolute() or '..' in path.parts
                    or not path.parts or (path.parts[0] not in prefixes and not (seed in (None, 'all') and info.filename in REPORT_FILES))
                    or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError('Unsafe or unexpected archive path.')
                if info.is_dir(): continue
                if seed == 'all' and path.parts[0].startswith('seed_'):
                    seeds.append(meta['prepared_sha256'])
                target = output.joinpath(*path.parts).resolve()
                if not target.is_relative_to(output): raise ValueError('Unsafe archive destination.')
                with bundle.open(info) as stream:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
                digest = h.hexdigest()
                if info.filename in files and files[info.filename][2] != digest:
                    raise ValueError('Conflicting seed ZIPs; upload only the latest ZIP for each seed.')
                if target.exists() and file_hash(target) != digest:
                    raise ValueError('Existing results differ; restore the latest ZIPs into a fresh runtime.')
                files[info.filename] = (archive, info, digest)
    prepared = files.get('shared/prepared.json')
    prepared_hash = prepared[2] if prepared else file_hash(output/'shared'/'prepared.json') if (output/'shared'/'prepared.json').exists() else None
    if seeds and (prepared_hash is None or any(h != prepared_hash for h in seeds)):
        raise ValueError('Upload the matching shared ZIP together with seed ZIPs.')
    missing_bytes = sum(info.file_size for name, (_, info, _) in files.items() if not (output/name).exists())
    if missing_bytes > shutil.disk_usage(output).free:
        raise OSError('Not enough local space to restore these results.')
    for name, (archive, info, _) in files.items():
        target = output/name
        if target.exists(): continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with zipfile.ZipFile(archive) as bundle, bundle.open(info) as source:
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as destination:
                    temporary = Path(destination.name)
                    shutil.copyfileobj(source, destination)
            os.replace(temporary, target)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
    return len(files)
