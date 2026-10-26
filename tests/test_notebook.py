import base64
import hashlib
import io
import json
import zipfile

import nbformat

from build_notebook import build, sources, DESTINATION, ROOT


def test_standalone_notebook_matches_source_and_has_no_saved_outputs():
    book = build(write=False)
    nbformat.validate(book)
    assert book == nbformat.read(DESTINATION,as_version=4)
    cells = [cell.source for cell in book.cells if cell.cell_type=='code']
    for source in cells:
        compile(source,'colab_cell','exec')
    assert all(not c.outputs and c.execution_count is None for c in book.cells if c.cell_type=='code')
    unpack = next(c for c in cells if 'PAYLOAD = ' in c)
    payload = unpack.split('PAYLOAD = "',1)[1].split('"',1)[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:
        bundled = {n:bundle.read(n) for n in bundle.namelist()}
    assert bundled == sources()
    for name, body in bundled.items():
        if name.endswith('.py'): compile(body,name,'exec')
    manifest = json.loads(bundled['manifest.json'])
    assert all(hashlib.sha256(bundled[n]).hexdigest()==h for n,h in manifest.items())
    joined = '\n'.join(cells)
    assert 'files.upload' not in joined and 'userdata.get' not in joined
    assert 'drive.mount' in joined and 'Download output ZIP' in joined
    assert 'HF_TOKEN = ""' in joined


def test_three_seed_dispatch_and_final_test_are_independent():
    book = build(write=False)
    source = next(c.source for c in book.cells if c.cell_type=='code' and 'for seed in PLAN[' in c.source)
    plan = json.loads((ROOT/'phi_pilot/plan.json').read_text(encoding='utf8'))
    calls = []
    exec(source,{'PLAN':plan,'command':lambda *a,**kw:calls.append((a,kw))})
    trains = [(args[2],kw['seed']) for args,kw in calls if args[0]=='train']
    assert trains == [(a,s) for s in (1,2,3) for a in ('mon','rep','quota','kl')]
    for args,kw in calls:
        if args[0]=='evaluate':
            assert args[-1] == ('test' if args[4]==300 else 'validation')
    assert len([1 for a,k in calls if a[0]=='evaluate' and a[-1]=='test'])==12


def test_frozen_primary_splits_and_upstream_controllers():
    folder=ROOT/'phi_pilot'
    data=json.loads((folder/'questions.json').read_text(encoding='utf8'))
    plan=json.loads((folder/'plan.json').read_text(encoding='utf8'))
    seen=set()
    for split,size in (('train',2000),('validation',256),('test',2000),('anchor_pool',2048)):
        assert len(data[split])==size
        keys={' '.join(r['question'].lower().split()) for r in data[split]}
        assert len(keys)==size and not seen.intersection(keys)
        seen.update(keys)
    for name in ('constraints.py','flipcon.py'):
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==plan['upstream_sha256'][name]
    assert plan['seeds']==[1,2,3] and plan['arms']==['mon','rep','quota','kl']
    assert plan['pi_budget_tokens']==400.86 and plan['context_tokens']==1538
    assert plan['micro_batch']==1 and plan['steps']==300
