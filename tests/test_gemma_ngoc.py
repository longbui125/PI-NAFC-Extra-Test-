"""CPU-only checks against actual Ngoc run arguments and untouched upstream source."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import nbformat
import pytest


def test_real_ipython_writefile_bytes_and_live_command_output(tmp_path, monkeypatch, capsys):
    import ast, io, os, subprocess, types
    from IPython.core.inputtransformer2 import TransformerManager
    from build_colab import build
    book=nbformat.read(build('gemma'),as_version=4)
    for cell in book.cells:
        if cell.cell_type!='code' or not cell.source.startswith('%%writefile'):
            continue
        name=cell.source.split('\n',1)[0].rsplit('/',1)[-1]
        transformed=TransformerManager().transform_cell(cell.source)
        body=ast.parse(transformed).body[0].value.args[2].value
        (tmp_path/name).write_bytes(body.encode('utf-8'))
    colab=types.ModuleType('google.colab');colab.files=types.SimpleNamespace()
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    ns=dict(ROOT=tmp_path,RESTORE_MATCHED_OUTPUT=False,os=os,sys=sys,subprocess=subprocess)
    setup=next(c.source for c in book.cells if c.cell_type=='code' and 'spec.loader.exec_module(exporter)' in c.source)
    exec(setup,ns)
    p=ns['PLAN']
    for name,expected in p['source_sha256'].items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==expected
    calls=[]
    class Process:
        stdout=io.StringIO('Downloading model...\nProcessed prompts: 20/2000\n')
        def wait(self):return 0
    def popen(*args,**kwargs):
        calls.append(kwargs);return Process()
    monkeypatch.setattr(subprocess,'Popen',popen)
    ns['command']('prepare')
    assert calls[0]['stdout']==subprocess.PIPE and calls[0]['stderr']==subprocess.STDOUT
    assert 'Processed prompts: 20/2000' in capsys.readouterr().out
    path=tmp_path/'build_anchors.py';path.write_bytes(path.read_bytes()+b'# unexpected edit\n')
    with pytest.raises(ValueError,match='pinned hash'):exec(setup,ns)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'src/gemma_ngoc'


def load_runner():
    spec = importlib.util.spec_from_file_location('gemma_matched_test', SOURCE/'matched_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(tmp_path, monkeypatch):
    module = load_runner()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'MODEL', tmp_path.parent/'gemma_ngoc_model')
    for path in SOURCE.iterdir():
        if path.is_file():
            (tmp_path/path.name).write_bytes(path.read_bytes())
    monkeypatch.syspath_prepend(str(tmp_path))
    return module


def test_new_plan_matches_ngoc_scale_and_preserves_upstream_text():
    p = load_runner().check_sources()
    assert p['seeds'] == [1, 2, 3] and len(p['arms']) == 5
    assert (p['max_tokens'], p['pi_budget'], p['reference_budget']) == (1536, 358, 440)
    assert (p['train_count'], p['test_count'], p['steps']) == (2000, 2000, 300)
    assert p['precision'] == 'bfloat16' and p['anchor_pool'] == 800 and p['anchor_noise_repeats'] == 1
    assert p['dataset']=='olympic' and p['train_dataset']=='AI-MO/NuminaMath-1.5' and p['test_dataset']=='KbsdJames/Omni-MATH'
    assert p['keep_steps'] == [50, 100, 150, 200] and p['test_draws'] == 4
    for name, digest in p['source_sha256'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest
    upstream = PROJECT.parent/'DATN-FPTU-rl-nafc/code/qwen-nafc'
    for name in ('common.py','constraints.py','train.py','flipcon.py','evaluate.py','prepare_buckets.py','build_anchors.py','fixflip.py'):
        if upstream.exists():
            expected=(upstream/name).read_text(encoding='utf-8')
            if name=='prepare_buckets.py':
                expected=expected.replace('choices=["gsm8k", "math500", "math"]','choices=["gsm8k", "math500", "math", "olympic"]',1)
            assert (SOURCE/name).read_text(encoding='utf-8') == expected


@pytest.mark.parametrize('arm', ['mon', 'rep', 'fix', 'kl', 'fg'])
def test_training_arguments_match_actual_ngoc_run_info(arm):
    m = load_runner(); args = m.train_args(arm, 1)
    parsed = {}; i = 0
    while i < len(args):
        key = args[i][2:]
        if key == 'two_sided':
            parsed[key] = True; i += 1
        else:
            parsed[key] = args[i+1]; i += 2
    actual = json.loads((PROJECT/'tests/fixtures/ngoc_pi_args.json').read_text(encoding='utf-8'))
    expected = dict(actual[arm if arm in actual else 'fix'])
    if arm in ('mon', 'rep'):
        expected.update(flip_mode='monitor' if arm == 'mon' else 'nafc_replay',
                        flip_gamma=.1, flip_eta=1.0 if arm == 'mon' else 10.0)
    for key, value in parsed.items():
        if key in ('model', 'output_dir', 'merged_dir'):
            continue
        assert type(expected[key])(value) == expected[key], key
    assert parsed['model'] == str(m.MODEL)
    assert parsed['output_dir'] == f'runs/lacPI{arm}H_s1'
    assert '--no_vllm' not in args


def test_gemma_notebook_is_standalone_matched_and_has_no_old_resume_payload():
    from build_colab import build
    book = nbformat.read(build('gemma'), as_version=4)
    nbformat.validate(book)
    p = load_runner().check_sources()
    embedded = {}
    for cell in book.cells:
        if cell.cell_type != 'code':
            continue
        assert cell.outputs == [] and cell.execution_count is None
        if cell.source.startswith('%%writefile'):
            line, body = cell.source.split('\n', 1)
            name = line.rsplit('/', 1)[-1]
            embedded[name] = body.encode('utf-8')
            if name.endswith('.py'): compile(body, name, 'exec')
        else:
            compile(cell.source, 'gemma_cell', 'exec')
    assert set(embedded) == set(p['source_sha256']) | {'plan.json'}
    for name, digest in p['source_sha256'].items():
        assert hashlib.sha256(embedded[name]).hexdigest() == digest
    assert book.metadata.nafc.profile == 'gemma_ngoc_matched'
    assert not book.metadata.nafc.historical_output_reused
    text='\n'.join(c.source for c in book.cells if c.cell_type == 'code')
    assert 'EMBEDDED_PROGRESS' not in text and 'drive.mount' not in text and 'userdata' not in text
    assert 'RESTORE_MATCHED_OUTPUT = False' in text
    train=next(c.source for c in book.cells if c.cell_type=='code' and 'for seed in PLAN[' in c.source)
    calls=[];exec(train,dict(PLAN=p,command=lambda *a:calls.append(a)))
    assert calls == [('train','--arm',a,'--seed',s) for s in [1,2,3] for a in m_arms()]


def m_arms():
    return ('mon','rep','fix','kl','fg')


def test_original_olympic_questions_are_preserved_and_anchors_are_disjoint(monkeypatch):
    spec=importlib.util.spec_from_file_location('gemma_input_test',SOURCE/'run_upstream.py')
    module=importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(SOURCE));spec.loader.exec_module(module)
    frozen=json.loads((SOURCE/'frozen_questions.json').read_text(encoding='utf-8'))
    assert module.load_split('olympic','train',2000) == frozen['train']
    assert module.load_split('olympic','test',2000) == frozen['test']
    assert module.load_split('olympic','train',None)==frozen['train']+frozen['anchor_pool']
    original=json.loads((PROJECT/'src/colab_splits.json').read_text(encoding='utf-8'))['splits']
    pairs=lambda rows:[{k:q[k] for k in ('question','answer')} for q in rows]
    assert frozen['train']==pairs(original['task_train'])
    assert frozen['test']==pairs(original['omni_test'])
    assert frozen['anchor_pool']==pairs(original['anchor_candidates'][:800])
    keys=[q['question'] for name in ('train','test','anchor_pool') for q in frozen[name]]
    assert len(keys)==len(set(keys))==4800
    calls=[];monkeypatch.setattr(module,'original_load_split',lambda *a:calls.append(a) or [])
    module.load_split('math','train',None)
    assert calls==[('math','train',None)]


def test_preparation_runs_only_one_cap_and_reuses_completed_stages(runner, monkeypatch):
    m=runner;p=m.check_sources();calls=[]
    monkeypatch.setattr(m,'initialize',lambda:p)
    def fake(script,args,log):
        calls.append((script,args))
        if script=='prepare_buckets.py':
            target=m.ROOT/args[args.index('--out')+1]
            m.write(target,{'items':[{}]*2000,'meta':{'model':str(m.MODEL),'max_len':1536,'dataset':'olympic'}})
        elif script=='evaluate.py':
            m.write(m.ROOT/'results/base.json',{'per_question':[]})
        elif script=='build_anchors.py':
            m.write(m.ROOT/'anchors.json',{'items':[{}]*128})
    monkeypatch.setattr(m,'execute',fake)
    m.prepare()
    assert [a[0] for a in calls]==['prepare_buckets.py']*3+['evaluate.py','build_anchors.py']
    assert all('--max_len' not in a or a[a.index('--max_len')+1]==1536 for _,a in calls)
    calls.clear();m.prepare();assert not calls
    path=m.ROOT/'buckets_test_B440.json'
    wrong=m.read(path);wrong['meta']['dataset']='math';m.write(path,wrong)
    with pytest.raises(ValueError,match='Wrong matched data partition'):m.prepare()


def test_export_restores_all_checkpoints_and_excludes_credentials_and_weights(runner,tmp_path,monkeypatch):
    m=runner;m.write(m.ROOT/'experiment_identity.json',{'verified':True})
    ck=m.ROOT/'runs/lacPIrepH_s3/checkpoint-300';ck.mkdir(parents=True)
    (ck/'optimizer.pt').write_bytes(b'optimizer')
    (m.ROOT/'model.safetensors').write_bytes(b'exclude weights')
    (m.ROOT/'.env').write_text('exclude credentials')
    archive=m.pack(tmp_path.parent/'gemma_matched_export.zip')
    with zipfile.ZipFile(archive) as z:
        assert 'runs/lacPIrepH_s3/checkpoint-300/optimizer.pt' in z.namelist()
        assert 'model.safetensors' not in z.namelist() and '.env' not in z.namelist()
    new=tmp_path/'restored';new.mkdir()
    (new/'plan.json').write_bytes((m.ROOT/'plan.json').read_bytes())
    for name in m.check_sources()['source_sha256']:
        (new/name).write_bytes((m.ROOT/name).read_bytes())
    monkeypatch.setattr(m,'ROOT',new);m.restore(archive)
    assert (new/'runs/lacPIrepH_s3/checkpoint-300/optimizer.pt').read_bytes()==b'optimizer'


@pytest.mark.parametrize('name',['../escape','/absolute','runs/../../escape','runs\\escape','hf_cache/token'])
def test_restore_rejects_unsafe_paths_before_writing(runner,tmp_path,name):
    m=runner;archive=tmp_path.parent/'unsafe_gemma.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(m.MANIFEST,json.dumps({'plan':m.check_sources()}))
        z.writestr(name,'unsafe')
    if '\\' in name:
        # Windows ZipInfo normalizes separators; construct the actual untrusted member name.
        archive.write_bytes(archive.read_bytes().replace(name.replace('\\','/').encode(),name.encode()))
    before=set(m.ROOT.rglob('*'))
    with pytest.raises(ValueError,match='Unsafe'):m.restore(archive)
    assert before==set(m.ROOT.rglob('*'))


def test_old_output_is_rejected_and_progress_inspection_is_correct(runner):
    old=PROJECT/'snapshots/nafc_gemma_math_3seeds_output.zip'
    if not old.exists():pytest.skip('User output is not distributed with the source checkout.')
    with pytest.raises(ValueError,match='not from'):runner.restore(old)
    with zipfile.ZipFile(old) as z:
        validation=json.loads(z.read('shared/reference_validation.json'))
        test=json.loads(z.read('shared/reference_math_test.json'))
        selection=json.loads(z.read('shared/anchor_selection.json'))
        assert len(validation['records'])//4==64 and len(test['records'])//8==208
        assert selection['tested']==336
        assert all(len(json.loads(z.read(f'shared/anchor_noise_{i}.json'))['records'])==256 for i in range(3))


def test_resume_chooses_last_complete_checkpoint_not_partial_one(runner):
    m=runner;folder=m.ROOT/'runs/test'
    for step in (25,50):
        ck=folder/f'checkpoint-{step}';ck.mkdir(parents=True)
        for name in ('adapter_model.safetensors','adapter_config.json','trainer_state.json','optimizer.pt',
                     'scheduler.pt','rng_state.pth','lambda_state.json','flip_state.json'):
            if step==50 and name=='optimizer.pt':continue
            (ck/name).write_bytes(b'state')
    assert m.latest_checkpoint(folder).name=='checkpoint-25'


def test_report_contains_base_five_arms_three_seeds_all_steps(runner,monkeypatch):
    m=runner;p=m.check_sources();monkeypatch.syspath_prepend(str(SOURCE))
    m.write(m.ROOT/'buckets_test_B440.json',{'meta':{'n':8},'items':[
        {'p':v,'bucket':0,'samples':[[True,100]]*8} for v in (1,1,0)]})
    def result(pattern):
        return {'decoding':{'n':4},'overall':{'n':3,'acc':100*sum(pattern)/3,'mean_len':100,
                 'boxed_rate':100,'truncated_rate':0},'per_question':[
                     {'acc':x,'bucket':0,'len':100,'boxed':1,'trunc':0} for x in pattern]}
    m.write(m.ROOT/'results/base.json',result([1,1,0]))
    for seed in p['seeds']:
        for arm in m.ARMS:
            for step in p['test_steps']:
                name=m.run_name(arm,seed)+(f'_step{step}' if step!=300 else '')
                m.write(m.ROOT/f'results/{name}.json',result([1,1 if arm=='rep' else 0,1]))
    m.report();text=(m.ROOT/'REPORT.md').read_text(encoding='utf-8')
    assert 'Base sampled accuracy:' in text
    assert all(f'| {step} | {seed} | {arm} |' in text for step in p['test_steps'] for seed in p['seeds'] for arm in m.ARMS)
    (m.ROOT/'results/lacPIfgH_s3.json').unlink()
    with pytest.raises(ValueError,match='All prespecified'):m.report()
