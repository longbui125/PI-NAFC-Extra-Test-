"""Gemma persists checkpoints on Drive while temporary model copies stay local."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import nbformat
import pytest

from build_colab import build
from test_gemma_ngoc import SOURCE, runner


def test_drive_mount_requires_token_and_runs_once_before_install(monkeypatch):
    book = nbformat.read(build('gemma'),as_version=4)
    access = next(c.source for c in book.cells if c.cell_type=='code' and 'HF_TOKEN = ""' in c.source)
    calls=[]
    module=types.ModuleType('google.colab')
    module.drive=types.SimpleNamespace(mount=calls.append)
    monkeypatch.setitem(sys.modules,'google.colab',module)
    monkeypatch.setenv('HF_TOKEN','')
    with pytest.raises(ValueError,match='Fill HF_TOKEN'):
        exec(access,{})
    assert not calls
    exec(access.replace('HF_TOKEN = ""','HF_TOKEN = "test-token-not-a-credential"',1),{})
    assert calls==['/content/drive']


def test_gemma_writes_sources_and_outputs_on_drive_but_keeps_weights_local():
    book=nbformat.read(build('gemma'),as_version=4)
    root='/content/drive/MyDrive/NAFC_Gemma_Olympiad_micro2'
    setup=next(c.source for c in book.cells if c.cell_type=='code' and '"pip", "install"' in c.source)
    assert f'ROOT = Path("{root}")' in setup
    for cell in book.cells:
        if cell.cell_type=='code' and cell.source.startswith('%%writefile'):
            assert cell.source.split('\n',1)[0].startswith('%%writefile '+root+'/')
    source=next(c.source.split('\n',1)[1] for c in book.cells if c.cell_type=='code'
                and c.source.startswith('%%writefile '+root+'/matched_run.py\n'))
    assert "MODEL = Path('/content/gemma_ngoc_model')" in source
    assert "str(MODEL.parent/'merged'/name)" in source
    assert "merged = MODEL.parent/'merged'/key" in source
    assert "ROOT.parent/'merged'" not in source


def test_new_runtime_resumes_drive_checkpoint_and_skips_finished_runs(runner,monkeypatch):
    # The same persisted folder is imported by a fresh module, as after a Colab VM replacement.
    plan=runner.check_sources()
    folder=runner.ROOT/'runs/lacPIrepH_s1/checkpoint-25';folder.mkdir(parents=True)
    for name in ('adapter_model.safetensors','adapter_config.json','trainer_state.json','optimizer.pt',
                 'scheduler.pt','rng_state.pth','lambda_state.json','flip_state.json'):
        (folder/name).write_bytes(b'persisted state')
    partial=folder.parent/'checkpoint-50';partial.mkdir()
    (partial/'adapter_model.safetensors').write_bytes(b'incomplete checkpoint')
    runner.write(runner.ROOT/'anchors.json',{'items':[{}]*128})
    for arm in ('mon','rep'):
        key=runner.run_name(arm,1)
        for step in [plan['steps'],*plan['keep_steps']]:
            name=key if step==plan['steps'] else f'{key}_step{step}'
            runner.write(runner.ROOT/f'results/{name}.json',{'saved':True})
    runner.write(runner.ROOT/'runs/lacPImonH_s1/run_info.json',{'finished':True})
    spec=importlib.util.spec_from_file_location('fresh_drive_runner',runner.ROOT/'matched_run.py')
    fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
    calls=[]
    monkeypatch.setattr(fresh,'initialize',lambda:plan)
    monkeypatch.setattr(fresh,'execute',lambda *a:calls.append(a))
    fresh.train_one('mon',1)
    assert not calls
    fresh.train_one('rep',1)
    assert len(calls)==1 and calls[0][0]=='train.py'
    args=calls[0][1]
    assert args[args.index('--resume')+1]==str(folder)
    assert args[args.index('--micro_batch')+1]=='2'
    assert args[args.index('--merged_dir')+1].startswith(str(Path('/content/merged')))


def test_existing_drive_progress_is_not_overwritten_by_zip_restore(runner,monkeypatch):
    book=nbformat.read(build('gemma'),as_version=4)
    runner.write(runner.ROOT/'experiment_identity.json',{'persisted':True})
    runner.write(runner.ROOT/'anchors.json',{'persisted anchors':True})
    before=(runner.ROOT/'anchors.json').read_bytes()
    module=types.ModuleType('google.colab')
    module.files=types.SimpleNamespace(upload=lambda:pytest.fail('Old ZIP requested over existing Drive progress'))
    monkeypatch.setitem(sys.modules,'google.colab',module)
    setup=next(c.source for c in book.cells if c.cell_type=='code' and 'spec.loader.exec_module(exporter)' in c.source)
    exec(setup.replace('exporter.check_process_cleanup()',''),dict(ROOT=runner.ROOT,RESTORE_MATCHED_OUTPUT=True,sys=sys))
    assert (runner.ROOT/'anchors.json').read_bytes()==before


def test_saved_micro1_preparation_is_compatible_but_training_is_not(runner,tmp_path):
    import zipfile
    old=json.loads((Path(__file__).parent/'fixtures/gemma_micro1_plan.json').read_text(encoding='utf-8'))
    assert runner.same_experiment(old,runner.check_sources())
    archive=tmp_path/'micro1.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('MATCHED_RUN.json',json.dumps({'plan':old}))
        z.writestr('anchors.json','saved model-specific preparation')
    runner.restore(archive)
    assert (runner.ROOT/'anchors.json').read_text()=='saved model-specific preparation'
    with zipfile.ZipFile(archive,'a') as z:
        z.writestr('runs/lacPImonH_s1/checkpoint-25/optimizer.pt',b'optimizer')
    with pytest.raises(ValueError,match='different micro-batch'):
        runner.restore(archive)


def test_phi_keeps_micro1_and_can_recognize_previous_phi_preparation(tmp_path):
    book=nbformat.read(build('phi'),as_version=4)
    for cell in book.cells:
        if cell.cell_type=='code' and cell.source.startswith('%%writefile'):
            header,body=cell.source.split('\n',1)
            (tmp_path/header.rsplit('/',1)[-1]).write_bytes(body.encode('utf-8'))
    spec=importlib.util.spec_from_file_location('phi_old_plan_compat',tmp_path/'matched_run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=json.loads((Path(__file__).parent/'fixtures/phi_micro1_plan.json').read_text(encoding='utf-8'))
    assert module.check_sources()['micro_batch']==1
    assert module.same_experiment(old,module.check_sources())


def test_actual_saved_preparation_skips_all_gpu_sampling_on_restart(runner,monkeypatch):
    archive=SOURCE.parents[1]/'snapshots/nafc_gemma_ngoc_matched_output.zip'
    if not archive.exists():
        pytest.skip('User preparation ZIP is local and is not published')
    runner.restore(archive)
    monkeypatch.setattr(runner,'MODEL','/content/gemma_ngoc_model')
    monkeypatch.setattr(runner,'initialize',runner.check_sources)
    monkeypatch.setattr(runner,'execute',lambda *a:pytest.fail('Completed preparation triggered GPU sampling'))
    monkeypatch.syspath_prepend(str(SOURCE))
    runner.prepare()
