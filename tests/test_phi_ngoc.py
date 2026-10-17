"""Phi shares frozen questions and algorithms, never another model's outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

import nbformat
import pytest

from build_colab import build

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def phi(tmp_path):
    book = nbformat.read(build('phi'), as_version=4)
    nbformat.validate(book)
    bodies = {}
    for cell in book.cells:
        if cell.cell_type != 'code':
            continue
        assert cell.outputs == [] and cell.execution_count is None
        if cell.source.startswith('%%writefile'):
            header, body = cell.source.split('\n', 1)
            name = header.rsplit('/', 1)[-1]
            assert header == '%%writefile /content/phi_ngoc/'+name
            bodies[name] = body.encode('utf-8')
            (tmp_path/name).write_bytes(bodies[name])
            if name.endswith('.py'):
                compile(body, name, 'exec')
        else:
            compile(cell.source, 'phi_cell', 'exec')
    spec = importlib.util.spec_from_file_location('matched_phi_test', tmp_path/'matched_run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return book, bodies, runner


def test_phi_matches_gemma_protocol_and_reuses_only_question_lists(phi):
    book, bodies, runner = phi
    plan = runner.check_sources()
    gemma = json.loads((PROJECT/'src/gemma_ngoc/plan.json').read_text(encoding='utf-8'))
    assert plan['model'] == 'microsoft/Phi-4-mini-instruct'
    assert plan['model_revision'] == 'cfbefacb99257ffa30c83adab238a50856ac3083'
    runtime_fields = ('model','model_revision','source_sha256','source_line_endings','vllm_training_context')
    assert {k:v for k,v in plan.items() if k not in runtime_fields} == {
        k:v for k,v in gemma.items() if k not in runtime_fields}
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256(bodies[name]).hexdigest() == digest
        if name not in ('matched_run.py', 'train.py'):
            assert bodies[name] == (PROJECT/'src/gemma_ngoc'/name).read_bytes()
    assert runner.MODEL.name == 'phi_ngoc_model'
    assert book.metadata.nafc.profile == 'phi_ngoc_matched'
    assert not book.metadata.nafc.historical_output_reused
    text = '\n'.join(c.source for c in book.cells if c.cell_type == 'code')
    assert all(s not in text for s in ('EMBEDDED_PROGRESS','HF_TOKEN','userdata','drive.mount','RCLONE'))
    assert not (runner.ROOT/'anchors.json').exists()


def test_phi_context_covers_task_and_anchor_prompts_without_truncation(phi):
    import ast
    _, bodies, _ = phi
    tree = ast.parse(bodies['train.py'].decode('utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(i for i,node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0],ast.Name) and node.targets[0].id == 'tokenizer')
    finish = next(i for i,node in enumerate(main.body) if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0],ast.Name) and node.targets[0].id == 'config')
    seen = []
    def encode(messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True)
        seen.append(messages)
        return list(range(messages[0]['length']))
    tok = types.SimpleNamespace(apply_chat_template=encode)
    scope = dict(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw:tok),
                 args=types.SimpleNamespace(model='Phi',max_completion_length=1536),
                 ds={'prompt':[[{'length':20}],[{'length':80}]]},
                 anchors=types.SimpleNamespace(questions=[120]), chat_messages=lambda q:[{'length':q}])
    exec(compile(ast.Module(body=main.body[start:finish],type_ignores=[]),'phi_context','exec'),scope)
    assert scope['context_limit'] == 120+1536 and len(seen) == 3
    config = main.body[finish].value
    assert next(k.value.id for k in config.keywords if k.arg == 'vllm_max_model_length') == 'context_limit'
    assert 'truncation=True' not in bodies['train.py'].decode('utf-8')


def test_phi_real_ipython_writefile_hash_normalization(phi, monkeypatch):
    import ast
    from IPython.core.inputtransformer2 import TransformerManager
    book, bodies, runner = phi
    for cell in book.cells:
        if cell.cell_type == 'code' and cell.source.startswith('%%writefile'):
            call = ast.parse(TransformerManager().transform_cell(cell.source)).body[0].value
            name = call.args[1].value.rsplit('/', 1)[-1]
            (runner.ROOT/name).write_bytes(call.args[2].value.encode('utf-8'))
    fake = types.ModuleType('google.colab'); fake.files = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'google.colab', fake)
    setup = next(c.source for c in book.cells if c.cell_type == 'code' and 'spec.loader.exec_module(exporter)' in c.source)
    ns = dict(ROOT=runner.ROOT, RESTORE_MATCHED_OUTPUT=False, sys=sys)
    # Linux process-group cleanup is exercised in the real Colab before any GPU work.
    exec(setup.replace('exporter.check_process_cleanup()', ''), ns)
    assert ns['PLAN'] == runner.check_sources()
    calls = []
    monkeypatch.setattr(ns['exporter'], 'run_logged', lambda *args: calls.append(args))
    ns['command']('prepare')
    assert calls[0][0][-1] == 'prepare'


def test_phi_rejects_gemma_zip_before_importing_any_model_output(phi, tmp_path):
    _, _, runner = phi
    plan = json.loads((PROJECT/'src/gemma_ngoc/plan.json').read_text(encoding='utf-8'))
    archive = tmp_path/'gemma.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('MATCHED_RUN.json', json.dumps({'plan':plan}))
        z.writestr('anchors.json', 'Gemma-only outputs')
    with pytest.raises(ValueError, match='Different experiment'):
        runner.restore(archive)
    assert not (runner.ROOT/'anchors.json').exists()


def test_phi_rejects_old_phi_preparation(phi):
    _, _, runner = phi
    with pytest.raises(ValueError, match='not from'):
        runner.restore(PROJECT/'snapshots/nafc_phi_olympic_notebook02_v6.zip')


def test_phi_own_output_roundtrip(phi, tmp_path):
    _, _, runner = phi
    runner.write(runner.ROOT/'anchors.json', {'items':['Phi-only result']})
    archive = runner.pack(tmp_path.parent/(tmp_path.name+'_phi.zip'))
    (runner.ROOT/'anchors.json').unlink()
    runner.restore(archive)
    assert runner.read(runner.ROOT/'anchors.json') == {'items':['Phi-only result']}


def test_phi_initialize_downloads_only_pinned_phi(phi, monkeypatch):
    import torch
    import huggingface_hub
    _, _, runner = phi
    plan = runner.check_sources()
    versions = {**plan['versions'], 'math-verify':'0.9.0', 'datasets':'4.8.5'}
    original_version = runner.importlib.metadata.version
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: versions[name] if name in versions else original_version(name))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'NVIDIA A100-SXM4-40GB')
    calls = []
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *a, **kw: calls.append((a,kw)))
    runner.initialize()
    assert calls[0][0] == ('microsoft/Phi-4-mini-instruct',)
    assert calls[0][1]['revision'] == plan['model_revision']
    runner.initialize()
    assert len(calls) == 2


def test_phi_runs_all_fifteen_prespecified_arms_and_downloads_local_output(phi):
    book, _, runner = phi
    calls = []
    train = next(c.source for c in book.cells if c.cell_type == 'code' and 'for seed in PLAN[' in c.source)
    exec(train, dict(PLAN=runner.check_sources(), command=lambda *a: calls.append(a)))
    assert calls == [('train','--arm',a,'--seed',s) for s in [1,2,3] for a in ('mon','rep','fix','kl','fg')]
    assert 'files.download(str(archive))' in book.cells[-1].source
    assert 'nafc_phi_ngoc_matched_output.zip' in book.cells[-1].source
    setup = next(c.source for c in book.cells if c.cell_type == 'code' and '"pip", "install"' in c.source)
    assert '"uninstall", "-y", "torchao"' in setup and 'CPU LoRA forward/backward passed' in setup
    assert 'AutoTokenizer.from_pretrained' in (runner.ROOT/'train.py').read_text(encoding='utf-8')


@pytest.mark.parametrize('arm', ['mon','rep','fix','kl','fg'])
def test_phi_training_flags_preserve_gemma_config(phi, arm):
    from test_gemma_ngoc import load_runner
    _, _, runner = phi
    args = runner.train_args(arm, 1)
    reference = load_runner().train_args(arm, 1)
    for key in ('--model','--merged_dir'):
        index = args.index(key)+1
        assert 'phi_ngoc_model' in args[index] if key == '--model' else args[index].endswith(runner.run_name(arm, 1))
        args[index] = reference[index]
    assert args == reference
