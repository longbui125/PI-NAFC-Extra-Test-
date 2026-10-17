"""Colab profile checks: objective equivalence, fair endpoints and standalone packaging."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0, str(ROOT))
import colab_pilot as pilot
import io_utils
import data
import engine
import model
from test_pilot import TinyLM, FakeTokenizer, FakeSampler, question, records, toy_inputs, toy_run


def config(): return io_utils.read(ROOT/'config_colab.json')


def test_memory_optimization_preserves_values_and_gradients():
    from colab_checks import check
    check()


def test_fixed_colab_profile_and_large_profile_are_separate():
    c = config(); pilot.validate(c)
    assert c['steps'] == 300 and c['train_count'] == 2000 and c['precision'] == 'fp16'
    assert c['prompts_per_step'] == c['group_size'] == 8
    assert c['steps']*c['prompts_per_step']*c['group_size'] == 19200
    assert c['test_steps'] == [50, 100, 150, 200, 300]
    assert 'fixed_replay_fraction' not in c
    assert c['max_new_tokens'] == 4096 and c['test_draws'] == 4 and c['seeds'] == [42, 43, 44]
    large = io_utils.read(ROOT/'config.json')
    assert large['steps'] == 625 and large['precision'] == 'bf16' and len(large['seeds']) == 5
    with pytest.raises(ValueError): pilot.validate(dict(c, initial_draws=4))
    with pytest.raises(ValueError): pilot.validate(dict(c, train_count=100))


def test_embedded_real_data_counts_and_config_validation():
    c = config()
    ds = pilot.prepare_data(c)
    pilot.validate_data(ds, c)
    assert len(ds['task_train']) == len(ds['omni_test']) == 2000
    with pytest.raises(ValueError, match='Frozen notebook data'):
        pilot.prepare_data(dict(c, split_seed=43))


def test_colab_partition_blocks_even_unscored_benchmark_rows(monkeypatch, tmp_path):
    import datasets
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    monkeypatch.setattr(data, 'numeric_gold', lambda a: a if a == '4' else None)
    def item(i):
        return {'problem': f'Olympic {i}', 'answer': '4', 'source': 'olympiads_ref',
            'question_type': 'math-word-problem', 'problem_is_valid': 'Yes', 'solution_is_valid': 'Yes', 'problem_type': 'Algebra'}
    numina = [item(i) for i in range(120)]
    omni = [dict(item(0), answer='proof')] + [item(i+200) for i in range(20)]
    monkeypatch.setattr(datasets, 'load_dataset', lambda name, **kw: omni if name == data.DATASETS['omni'] else numina)
    c = config(); c.update(steps=12, train_count=48, validation_count=8, anchor_pool=48, test_count=8)
    first = pilot.prepare_data(c)
    assert first == pilot.prepare_data(c)
    assert len(first['task_train']) == 48 and len(first['anchor_candidates']) == 48
    assert all(q['question'] != 'Olympic 0' for k, rr in first.items() if k != 'provenance' for q in rr)
    assert first['provenance']['evaluation_domains'] == ['omni']


@pytest.mark.parametrize('stable_count,expected', [(3, 0), (5, 0), (6, 0), (8, 8)])
def test_declared_anchor_range_and_progress_resume(tmp_path, stable_count, expected):
    c = config(); c.update(anchor_count=8, anchor_minimum=8, anchor_selection_batch=4, anchor_pool=8)
    pool = [question(i) for i in range(8)]
    calls = []
    def sampler(qs, n, seed):
        calls.append(seed)
        rr = records(qs, n)
        for r in rr: r['correct'] = int(r['id'].rsplit('-', 1)[1]) < stable_count
        return rr
    chosen = pilot.select_anchors(c, {'anchor_candidates': pool}, sampler, tmp_path)
    assert len(chosen) == expected and all(q['base_first_correct'] for q in chosen)
    assert len(pilot.select_anchors(c, {'anchor_candidates': pool}, sampler, tmp_path)) == expected
    assert len(calls) == 2


def test_engine_colab_endpoint_and_checkpoint_cadence(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'OUT', tmp_path)
    c = config(); c.update(steps=2, prompts_per_step=2, warmup_steps=1, checkpoint_every=1, precision='bf16', sample_progress=False, test_draws=1, test_do_sample=False, long_evaluation=False)
    qs = {'task_train': [question(i, 'numina') for i in range(4)],
          'omni_test': [question(i, 'omni') for i in range(2)],
          'provenance': {'task_domain': 'numina', 'evaluation_domains': ['omni']}}
    bundle = {'anchors': [dict(question(i+10), base_first_correct=True) for i in range(4)],
        'repair_idx': [0, 1], 'held_idx': [2, 3], 'eps': .02, 'budget': 2}
    settings = []
    class Sampler:
        def __init__(self, m, t, cfg): self.c = cfg; settings.append(cfg)
        def __call__(self, questions, n, seed): return records(questions, n)
    engine.train_arm(c, qs, bundle, 'grpo_pi', loader=lambda *a, **kw: (TinyLM(), FakeTokenizer()), sampler_factory=Sampler)
    rr = io_utils.read(tmp_path/'grpo_pi'/'final_omni_test.json')['records']
    assert len(rr) == 2 and all(r['sample_index'] == 0 for r in rr)
    assert settings[-1]['do_sample'] is False
    assert not (tmp_path/'grpo_pi'/'long_omni_test.json').exists()
    assert io_utils.read(tmp_path/'grpo_pi'/'complete.json')['step'] == 2


@pytest.mark.parametrize("domain", ["omni", "math"])
def test_colab_report_keeps_caps_and_rejects_partial_arms(tmp_path, monkeypatch, domain):
    import colab_report
    monkeypatch.setattr(pilot, 'OUT', tmp_path)
    shared = tmp_path/'shared'; shared.mkdir()
    qs = [question(i, domain) for i in range(6)]
    c = io_utils.read(ROOT/('config_gemma_colab.json' if domain == 'math' else 'config_colab.json'))
    io_utils.write(shared/'splits.json', {f'{domain}_test': qs, 'provenance': {'evaluation_domains': [domain]}})
    io_utils.write(shared/'prepared.json', {'anchors': [question(i) for i in range(128)], 'eps': .02, 'budget': 100})
    io_utils.write(shared/'config.json', c)
    identity = {'config': c, 'protocol': pilot.PROTOCOL}
    io_utils.write(shared/'identity.json', identity)
    def rr(pattern, n):
        values = records(qs, n)
        for i, correct in enumerate(pattern):
            for r in values[i*n:(i+1)*n]: r.update(correct=bool(correct), capped=True)
        return {'records': values}
    for prefix in ('', 'long_'):
        io_utils.write(shared/f'reference_{prefix}{domain}_test.json', rr([1, 1, 1, 1, 0, 0], 8))
        io_utils.write(shared/f'reference_{prefix}{domain}_noise.json', rr([1, 1, 1, 1, 0, 0], 4))
    for seed in c['seeds']:
        root = tmp_path/f'seed_{seed}'; root.mkdir()
        io_utils.write(root/'identity.json', dict(identity, config=dict(c, seed=seed)))
        for arm, pattern in zip(pilot.ARMS, ([1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 1, 0])):
            folder = root/arm; folder.mkdir()
            for step in c['test_steps']:
                values = [1, 0, 0, 0, 0, 1] if arm == 'grpo_pi' and step == 50 else pattern
                io_utils.write(engine.test_path(folder, step, c['steps'], domain), rr(values, 4))
            io_utils.write(folder/f'long_{domain}_test.json', rr(pattern, 4))
            io_utils.write(folder/'complete.json', {'anchor_prompts': int(arm.endswith('nafc')), 'step': c['steps'], 'test_steps': c['test_steps'], 'optimizer_updates': c['steps']})
            io_utils.write(folder/'audit_metrics.json', [{'step': 0, 'lam': .2}])
    result = pilot.report()
    assert result['questions'] == 6 and result['seeds'] == c['seeds']
    primary = result['endpoints']['primary']
    assert primary['unchanged_noise']['correct_to_low_success'] == 0
    assert primary['seeds']['42']['grpo_pi']['retention']['correct_to_low_success'] == 2
    assert primary['seeds']['42']['grpo_pi_nafc']['retention']['correct_to_low_success'] == 1
    assert primary['seeds']['42']['grpo_pi_nafc']['capped'] == 1
    diff = primary['comparisons']['grpo_pi']['flip_rate']
    assert diff['mean'] == -.25 and diff['seed_values'] == [-.25]*len(c['seeds'])
    assert (tmp_path/'overview.png').stat().st_size > 0
    assert (tmp_path/'trajectory.png').stat().st_size > 0
    assert list(result['trajectory']) == ['50', '100', '150', '200', '300']
    assert result['trajectory']['50']['seeds']['42']['grpo_pi']['retention']['correct_to_low_success'] == 3
    assert result['trajectory']['300'] == primary
    missing = tmp_path/'seed_42'/'grpo_pi'/f'step_100_{domain}_test.json'
    saved = io_utils.read(missing); missing.unlink()
    with pytest.raises(ValueError, match='Missing prespecified'): pilot.report()
    io_utils.write(missing, saved)
    for prefix in ('', 'long_'):
        io_utils.write(shared/f'reference_{prefix}{domain}_test.json', rr([0]*6, 8))
    empty = pilot.report()
    assert empty['endpoints']['primary']['comparisons']['grpo_pi']['flip_rate']['mean'] is None
    assert any('initially stable' in w for w in empty['warnings'])
    (tmp_path/'seed_44'/'grpo_pi_nafc'/'complete.json').unlink()
    with pytest.raises(ValueError, match='All prespecified'): pilot.report()


def test_expanded_profile_rejects_smaller_anchors_and_cherry_picked_seeds():
    c = config()
    assert c['anchor_count'] == c['anchor_minimum'] == 128
    assert pilot.ARMS == ('grpo_pi', 'grpo_pi_nafc') and c['long_eval_tokens'] == 8192
    with pytest.raises(ValueError, match='full declared'): pilot.validate(dict(c, anchor_minimum=64))
    with pytest.raises(ValueError, match='prespecified'): pilot.validate(dict(c, seeds=[42, 44]))
    with pytest.raises(ValueError, match='four-draw'): pilot.validate(dict(c, test_draws=1))
    for steps in ([100, 50, 300], [50, 50, 300], [50, 100], [0, 300], [50, 301]):
        with pytest.raises(ValueError, match='milestones'): pilot.validate(dict(c, test_steps=steps))


def test_colab_train_seeds_are_independent_and_outputs_are_isolated(tmp_path, monkeypatch):
    c = config(); shared = tmp_path/'shared'; shared.mkdir()
    ds = {'task_train': [question(0)]}
    def identity(cfg, data, reference):
        return {'config': cfg, 'data': data, 'reference': reference}
    monkeypatch.setattr(pilot, 'OUT', tmp_path)
    monkeypatch.setattr(pilot, 'hardware', lambda c: None)
    monkeypatch.setattr(pilot, 'base_path', lambda c: 'same-public-reference')
    monkeypatch.setattr(pilot, 'checkpoint_identity', lambda p: {'weights': 'same'})
    monkeypatch.setattr(pilot, 'identity', identity)
    io_utils.write(shared/'splits.json', ds)
    io_utils.write(shared/'prepared.json', {'identity_sha256': io_utils.digest(identity(c, ds, {'weights': 'same'}))})
    calls = []
    monkeypatch.setattr(engine, 'OUT', tmp_path)
    monkeypatch.setattr(engine, 'train_arm', lambda cfg, data, bundle, arm: calls.append((cfg, engine.OUT, arm)))
    for seed in c['seeds']:
        for arm in pilot.ARMS: pilot.train(c, arm, seed)
    assert len(calls) == 6
    assert {x[0]['seed'] for x in calls} == {42, 43, 44}
    assert all(cfg['reference_checkpoint'] == 'same-public-reference' for cfg, _, _ in calls)
    assert all(folder == tmp_path/f"seed_{cfg['seed']}" for cfg, folder, _ in calls)
    assert c['seed'] == 42
    with pytest.raises(ValueError, match='configuration changed'):
        pilot.train(dict(c, steps=301), pilot.ARMS[0], 42)


def test_repo_sized_task_schedule_repeats_without_dropping_groups():
    c = config()
    qs = [question(i, 'numina') for i in range(c['train_count'])]
    batches = data.schedule(qs, c['steps'], c['prompts_per_step'], c['seed'])
    assert len(batches) == 300 and all(len(b) == 8 for b in batches)
    assert all(len({q['id'] for q in b}) == 8 for b in batches)
    assert {q['id'] for b in batches[:250] for q in b} == {q['id'] for q in qs}
    assert sum(len(b) for b in batches[250:]) == 400
    assert batches == data.schedule(qs, c['steps'], 8, c['seed'])


def test_midpoint_tests_preserve_training_and_resume_exact_weights(tmp_path, monkeypatch):
    c, ds, bundle = toy_inputs()
    c.update(steps=4, checkpoint_every=3, validation_steps=[], test_steps=[],
             test_draws=4, test_do_sample=True, long_evaluation=True)
    monkeypatch.setattr(FakeSampler, 'bad_anchors', True)
    monkeypatch.setattr(FakeSampler, 'fail_step', None)
    plain = tmp_path/'plain'; plain.mkdir(); monkeypatch.setattr(engine, 'OUT', plain)
    toy_run(c, ds, bundle, 'grpo_pi_nafc')
    expected = torch.load(plain/'grpo_pi_nafc'/'resume.pt', weights_only=True)
    c = dict(c, test_steps=[2, 3, 4])
    run = tmp_path/'milestones'; run.mkdir(); monkeypatch.setattr(engine, 'OUT', run)
    original = engine.evaluate
    def interrupt(sampler, questions, n, seed, path):
        if path.name == 'step_2_omni_test.json':
            raise RuntimeError('Interrupted midpoint test')
        return original(sampler, questions, n, seed, path)
    monkeypatch.setattr(engine, 'evaluate', interrupt)
    with pytest.raises(RuntimeError, match='Interrupted midpoint'):
        toy_run(c, ds, bundle, 'grpo_pi_nafc')
    folder = run/'grpo_pi_nafc'
    saved = torch.load(folder/'resume.pt', weights_only=True)
    assert saved['state']['step'] == 2
    assert not (folder/'complete.json').exists()
    adapter = torch.load(folder/'adapters'/'step_2'/'tiny.pt', weights_only=True)
    for name, value in saved['parameters'].items():
        torch.testing.assert_close(value, adapter[name], rtol=0, atol=0)
    monkeypatch.setattr(engine, 'evaluate', original)
    toy_run(c, ds, bundle, 'grpo_pi_nafc')
    resumed = torch.load(folder/'resume.pt', weights_only=True)
    for name in expected['parameters']:
        torch.testing.assert_close(resumed['parameters'][name], expected['parameters'][name], rtol=0, atol=0)
    for name in ('pi', 'nafc', 'replay_ptr'):
        assert resumed['state'][name] == expected['state'][name]
    assert resumed['scheduler'] == expected['scheduler']
    for index, state in expected['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(resumed['optimizer']['state'][index][name], value, rtol=0, atol=0)
    assert io_utils.read(folder/'complete.json')['test_steps'] == [2, 3, 4]
    for step in c['test_steps']:
        for domain in ('numina', 'omni'):
            assert len(io_utils.read(engine.test_path(folder, step, 4, domain))['records']) == 8
    assert (folder/'long_omni_test.json').exists()
    assert not (folder/'step_2_long_omni_test.json').exists()


@pytest.mark.parametrize('profile,expected_questions', [('phi', 32)])
def test_two_notebooks_preserve_saved_sources_and_embed_valid_progress(profile, expected_questions):
    import ast, base64, hashlib, json, zipfile
    import nbformat
    from build_colab import build, load_progress, COMMON
    archive, meta, records, origins = load_progress(profile)
    book = nbformat.read(build(profile), as_version=4); nbformat.validate(book)
    actual = {}
    for cell in book.cells:
        if cell.cell_type != 'code': continue
        assert cell.execution_count is None and cell.outputs == []
        if cell.source.startswith('%%writefile'):
            first, body = cell.source.split('\n', 1)
            name = first.rsplit('/', 1)[-1]; actual[name] = body
            if name.endswith('.py'): compile(body, name, 'exec')
        else:
            compile(cell.source, 'notebook', 'exec')
        assert not any(term in cell.source for term in ('files.upload', 'userdata', 'RCLONE', 'drive.mount', 'getpass'))
    c = meta['identity']['config']
    assert set(actual) == set(COMMON) | {'config_colab.json', c['data_file']}
    assert json.loads(actual['config_colab.json']) == c
    assert c['seeds'] == book.metadata.nafc.assigned_seeds == [42, 43, 44]
    assert book.metadata.nafc.saved_validation_questions == expected_questions
    for name, expected in meta['identity']['code'].items():
        assert hashlib.sha256(actual[name].encode()).hexdigest() == expected
    payload_cell = next(c.source for c in book.cells if c.cell_type == 'code' and 'EMBEDDED_PROGRESS = (' in c.source)
    namespace = {}; exec(payload_cell, namespace)
    assert base64.b64decode(namespace['EMBEDDED_PROGRESS'], validate=True) == archive.read_bytes()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == book.metadata.nafc.embedded_progress_sha256


def extracted_notebook(profile, tmp_path):
    import nbformat
    from build_colab import build
    book = nbformat.read(build(profile), as_version=4)
    root = tmp_path/'sources'; root.mkdir()
    for cell in book.cells:
        if cell.cell_type == 'code' and cell.source.startswith('%%writefile'):
            first, body = cell.source.split('\n', 1)
            (root/first.rsplit('/',1)[-1]).write_bytes(body.encode('utf-8'))
    payload = next(c.source for c in book.cells if c.cell_type == 'code' and 'EMBEDDED_PROGRESS = (' in c.source)
    namespace = {}; exec(payload, namespace)
    return book, root, namespace['EMBEDDED_PROGRESS']


@pytest.mark.parametrize('profile,expected_questions', [('phi', 32)])
def test_embedded_restore_is_automatic_and_preserves_newer_local_progress(tmp_path, monkeypatch, profile, expected_questions):
    import importlib.metadata, os, subprocess, types
    import huggingface_hub
    from build_colab import load_progress
    book, root, payload = extracted_notebook(profile, tmp_path)
    _, meta, _, _ = load_progress(profile)
    versions = meta['identity']['versions']
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: versions[name])
    monkeypatch.setenv('HF_TOKEN', 'test-token')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    fake = types.ModuleType('google.colab'); fake.files = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'google.colab', fake)
    checks, downloads = [], []
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: checks.append(a))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *a, **kw: downloads.append(kw))
    namespace = dict(ROOT=root, OUTPUT=tmp_path/'out', ARCHIVES=tmp_path/'archives', EMBEDDED_PROGRESS=payload,
        os=os, sys=sys, subprocess=subprocess, importlib=importlib)
    namespace['OUTPUT'].mkdir(); namespace['ARCHIVES'].mkdir()
    cell = next(c.source for c in book.cells if c.cell_type == 'code' and 'Restored saved files:' in c.source)
    exec(cell, namespace)
    path = namespace['OUTPUT']/'shared'/'reference_validation.json'
    state = io_utils.read(path)
    assert len(state['records'])//4 == expected_questions
    assert not (namespace['OUTPUT']/'shared'/'prepared.json').exists()
    state['local_progress_marker'] = 'do not overwrite'; io_utils.write(path, state)
    exec(cell, namespace)
    assert io_utils.read(path)['local_progress_marker'] == 'do not overwrite'
    assert len(checks) == 2 and len(downloads) == 2*int(profile == 'gemma')


def test_restore_rejects_changed_source_before_unpacked_results(tmp_path, monkeypatch):
    import importlib, os, subprocess, types
    book, root, payload = extracted_notebook('phi', tmp_path)
    (root/'model.py').write_text('changed source', encoding='utf-8')
    colab = types.ModuleType('google.colab'); colab.files = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    output=tmp_path/'out'; output.mkdir()
    archives=tmp_path/'archives'; archives.mkdir()
    ns=dict(ROOT=root, OUTPUT=output, ARCHIVES=archives, EMBEDDED_PROGRESS=payload,
        os=os, sys=sys, subprocess=subprocess, importlib=importlib)
    cell=next(c.source for c in book.cells if c.cell_type=='code' and 'Restored saved files:' in c.source)
    with pytest.raises(ValueError, match='source changed: model.py'): exec(cell, ns)
    assert not list(output.iterdir())


def test_gemma_has_a_token_cell_and_phi_requires_no_token(monkeypatch):
    import nbformat, os
    from build_colab import build
    gemma=nbformat.read(build('gemma'),as_version=4)
    cell=next(c.source for c in gemma.cells if c.cell_type=='code' and 'HF_TOKEN = ""' in c.source)
    monkeypatch.delenv('HF_TOKEN',raising=False)
    with pytest.raises(ValueError,match='Fill HF_TOKEN'):exec(cell,{})
    filled=cell.replace('HF_TOKEN = ""','HF_TOKEN = "test-token-not-a-credential"',1)
    exec(filled,{})
    assert os.environ['HF_TOKEN']=='test-token-not-a-credential'
    phi=nbformat.read(build('phi'),as_version=4)
    assert not any('HF_TOKEN = ""' in c.source for c in phi.cells)


def test_builder_refuses_to_discard_different_phi_progress(tmp_path, monkeypatch):
    import json, zipfile
    import build_colab
    from shutil import copyfile
    saved = tmp_path/'snapshots'; saved.mkdir()
    names = ('nafc_phi_olympic_notebook02_v6.zip', 'nafc_phi_olympic_notebook03_v6.zip')
    copyfile(ROOT.parent/'snapshots'/names[0], saved/names[0])
    with zipfile.ZipFile(ROOT.parent/'snapshots'/names[1]) as original, zipfile.ZipFile(saved/names[1], 'w') as changed:
        for name in original.namelist():
            body = original.read(name)
            if name == 'shared/reference_validation.json':
                data = json.loads(body)
                data['records'][0]['correct'] = not data['records'][0]['correct']
                body = json.dumps(data).encode()
            changed.writestr(name, body)
    monkeypatch.setattr(build_colab, 'ROOT', tmp_path/'src')
    with pytest.raises(ValueError, match='Phi saved outputs differ'):
        build_colab.load_progress('phi')


def test_both_models_run_all_three_paired_seeds_and_keep_inconclusive_diagnostic(tmp_path):
    import nbformat
    from build_colab import build
    for profile in ('phi',):
        book=nbformat.read(build(profile),as_version=4)
        prep=next(c.source for c in book.cells if c.cell_type=='code' and 'command("prepare")' in c.source)
        train=next(c.source for c in book.cells if c.cell_type=='code' and 'for seed in ASSIGNED_SEEDS:' in c.source)
        output=tmp_path/profile; io_utils.write(output/'shared'/'prepared.json',{})
        calls=[]
        ns=dict(OUTPUT=output,ASSIGNED_SEEDS=[42,43,44],CONFIG={'seed':42},command=lambda *a:calls.append(a))
        exec(prep,ns);exec(train,ns)
        assert calls==[('prepare',)]+[('train','--seed',str(seed),'--arm',arm) for seed in [42,43,44] for arm in pilot.ARMS]
        calls.clear();(output/'shared'/'prepared.json').unlink()
        exec(train,ns);assert not calls


def test_end_of_notebook_download_button_exports_full_output_on_click(tmp_path, monkeypatch):
    import types, zipfile, nbformat, manual_artifacts, IPython.display
    from build_colab import build
    from test_manual_artifacts import fixture
    fixture(tmp_path/'output', {'seeds':[42,43,44]})
    io_utils.write(tmp_path/'output'/'seed_44'/'grpo_pi_nafc'/'resume.pt',{'optimizer':1})
    class Button:
        disabled=False
        def __init__(self,**kw):pass
        def on_click(self,callback):self.callback=callback
    class Output:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def clear_output(self,**kw):pass
    widgets=types.ModuleType('ipywidgets');widgets.Button=Button;widgets.Output=Output;widgets.Layout=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'ipywidgets',widgets)
    monkeypatch.setattr(IPython.display,'display',lambda *args:None)
    downloads=[]
    ns=dict(OUTPUT=tmp_path/'output',ARCHIVES=tmp_path/'archives',SLUG='phi_olympic',pack=manual_artifacts.pack,files=SimpleNamespace(download=downloads.append))
    book=nbformat.read(build('phi'),as_version=4)
    exec(book.cells[-1].source,ns);assert not downloads
    button=ns['download_button'];button.callback(button)
    assert len(downloads)==1 and not button.disabled
    with zipfile.ZipFile(downloads[0]) as bundle:
        assert 'shared/prepared.json' in bundle.namelist()
        assert 'seed_44/grpo_pi_nafc/resume.pt' in bundle.namelist()


def test_gemma_data_preserves_all_topics_and_answer_types(monkeypatch):
    import scoring
    c = io_utils.read(ROOT/'config_gemma_colab.json'); pilot.validate(c)
    ds = pilot.prepare_data(c); pilot.validate_data(ds, c)
    assert len(ds['task_train']) == len(ds['math_test']) == 2000
    assert all(len({q['topic'] for q in ds[name]}) == 7 for name in ('task_train', 'math_test', 'anchor_candidates'))
    assert ds['provenance']['excluded_missing_or_empty_gold'] == {'train': [5341, 5343], 'test': []}
    assert scoring.math_gold(r'Final $\boxed 2$.') == '2'
    assert scoring.math_gold(r'Final $\boxed{\frac{1}{2}}$.') == r'\frac{1}{2}'
    assert any(any(ch.isalpha() for ch in q['answer']) for q in ds['math_test'])
    seen = []
    monkeypatch.setattr(scoring, '_bounded', lambda request: seen.append(request) or True)
    assert scoring.math_is_correct(r'\boxed{x+1}', '1+x')
    assert seen == [{'op': 'verify_math', 'gold': '1+x', 'pred': 'x+1'}]
    assert not scoring.math_is_correct('no boxed answer', '1+x')
    with pytest.raises(ValueError, match='pinned'):
        pilot.validate(dict(c, base_model='Qwen/Qwen2.5-1.5B-Instruct'))
    with pytest.raises(ValueError, match='context'):
        pilot.validate(dict(c, max_prompt_tokens=c['context_length']))


def test_gemma_generation_skips_dynamo_and_preserves_lora_input_gradient_hook(monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3TextConfig, Gemma3ForCausalLM, CompileConfig
    base = Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=13, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, sliding_window=16, sliding_window_pattern=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=12))
    class Tokenizer(FakeTokenizer):
        pad_token_id = 0
        eos_token_id = 12
        def decode(self, *args, **kwargs): return r'\boxed{4}'
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: base)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    c = io_utils.read(ROOT/'config_gemma_colab.json')
    c.update(lora_rank=2, max_prompt_tokens=8, max_new_tokens=3, long_eval_tokens=6, generation_batch=2)
    loaded, tokenizer = model.load_model(c, 'unused', trainable=True)
    assert loaded.generation_config.disable_compile is True
    assert base.get_input_embeddings()._forward_hooks
    loaded._pilot_dtype = None  # CPU regression test; actual A100 execution still uses fp16.
    loaded.generation_config.compile_config = CompileConfig()
    loaded.generation_config.compile_config._compile_all_devices = True
    def compiled(*args, **kwargs): raise AssertionError('Gemma must not enter Dynamo generation.')
    monkeypatch.setattr(base, 'get_compiled_call', compiled)
    assert model.Sampler(loaded, tokenizer, c, scorer=lambda *a: True)([question(0)], 2, 123)
    loaded.train(); loaded.zero_grad(set_to_none=True)
    (-model.token_logps(loaded, [1, 2], [3, 4]).mean()).backward()
    gradients = [p.grad for p in loaded.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
