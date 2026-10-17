import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from test_gemma_ngoc import SOURCE, load_runner, runner


@pytest.fixture
def extension(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    spec = importlib.util.spec_from_file_location('extension_test', SOURCE/'anchor_extension.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'MODEL', tmp_path/'model')
    p = load_runner().check_sources()
    p.update(anchor_initial_pool=4, anchor_pool=8, anchor_count=4, anchor_batch_size=2, train_count=2)
    monkeypatch.setattr(m, 'check_sources', lambda: p)
    monkeypatch.setattr(m, 'is_correct', lambda text, answer: text == 'correct')
    rows = [{'question': f'q{i}', 'answer': f'a{i}'} for i in range(8)]
    m.write(tmp_path/'frozen_questions.json', {'train': [{'question': 'train', 'answer': '1'}],
                                            'test': [{'question': 'test', 'answer': '2'}], 'anchor_pool': rows})
    m.write(tmp_path/'experiment_identity.json', {'plan': p})
    items = [{**r, 'p_select': 1.0, 'base_first_correct': True, 'base_response': 'correct',
              'base_response_ids': [1, 2], 'base_response_len': 2, 'p_check': 1.0,
              'check_first_correct': True} for r in rows[:2]]
    m.write(tmp_path/'anchors.json', {'meta': {'model': str(m.MODEL), 'dataset': 'olympic', 'train_limit': 2,
            'pool': 4, 'n_anchor': 2, 'candidates': 2, 'pool_base_acc': .5, 'k_select': 8, 'k_check': 4,
            'hi': .75, 'lo': .25, 'max_len': 1536, 'temperature': 1.0}, 'items': items})
    return m


def response(rows, n, success=True):
    return [SimpleNamespace(outputs=[SimpleNamespace(text='correct' if success else 'wrong', token_ids=[1, 2])
                                    for _ in range(n)]) for row in rows]


def test_reuses_initial_qualifiers_scores_only_new_rows_and_recalibrates(extension):
    m = extension; calls = []
    def generate(rows, n, seed):
        calls.append(([r['question'] for r in rows], n, seed))
        return response(rows, n)
    m.build(generate)
    result = m.read(m.ROOT/'anchors.json')
    assert calls[:2] == [(['q4', 'q5'], 8, 0), (['q6', 'q7'], 8, 0)]
    assert len(calls[2][0]) == 4 and calls[2][1:] == (4, 1)
    assert result['meta']['pool'] == 8 and result['meta']['n_anchor'] == 4
    assert result['meta']['reused_qualified'] == 2 and result['meta']['pool_base_acc'] == .75
    assert result['meta']['noise_robust'] == 0
    assert len(m.read(m.ROOT/'results/anchor_extension/anchors_initial.json')['items']) == 2
    m.build(lambda *a: pytest.fail('Completed anchors were regenerated.'))


def test_interrupted_extension_resumes_after_saved_batch(extension):
    m = extension; calls = []
    def interrupted(rows, n, seed):
        calls.append([r['question'] for r in rows])
        if len(calls) == 2:
            raise RuntimeError('simulated disconnect')
        return response(rows, n)
    with pytest.raises(RuntimeError, match='simulated disconnect'):
        m.build(interrupted)
    assert len(m.read(m.ROOT/'anchors.json')['items']) == 2
    resumed = []
    def generate(rows, n, seed):
        resumed.append(([r['question'] for r in rows], n))
        return response(rows, n)
    m.build(generate)
    assert resumed[0] == (['q6', 'q7'], 8)
    assert len(resumed) == 2


def test_exhausted_bank_does_not_lower_threshold_or_repeat_completed_sampling(extension):
    m = extension
    with pytest.raises(RuntimeError, match='Found 2/4'):
        m.build(lambda rows, n, seed: response(rows, n, False))
    assert len(m.read(m.ROOT/'anchors.json')['items']) == 2
    with pytest.raises(RuntimeError, match='Found 2/4'):
        m.build(lambda *a: pytest.fail('Exhausted bank was regenerated.'))


def test_rejects_changed_cache_identity_and_overlapping_data(extension):
    m = extension
    m.write(m.ROOT/'results/anchor_extension/selection.json', {'identity': {'wrong': True}})
    with pytest.raises(ValueError, match='different experiment'):
        m.build(lambda *a: pytest.fail('Invalid cache triggered GPU work.'))
    frozen = m.read(m.ROOT/'frozen_questions.json')
    frozen['anchor_pool'][0]['question'] = ' TRAIN '
    m.write(m.ROOT/'frozen_questions.json', frozen)
    with pytest.raises(ValueError, match='disjoint'):
        m.build(lambda *a: pytest.fail('Overlapping data triggered GPU work.'))


def test_incomplete_calibration_keeps_selection_cache_and_original_anchor_file(extension):
    m = extension
    def generate(rows, n, seed):
        return response(rows, n) if n == 8 else []
    with pytest.raises(ValueError, match='Incomplete calibration'):
        m.build(generate)
    assert len(m.read(m.ROOT/'anchors.json')['items']) == 2
    assert len(m.read(m.ROOT/'results/anchor_extension/selection.json')['scored']) == 4


def test_real_user_zip_restores_66_anchors_without_modifying_saved_outputs(runner, monkeypatch):
    archive = SOURCE.parents[1]/'snapshots/nafc_gemma_ngoc_matched_output.zip'
    if not archive.exists():
        pytest.skip('User output is local and is not published.')
    runner.restore(archive)
    with zipfile.ZipFile(archive) as bundle:
        for name in ('anchors.json', 'results/base.json', 'buckets_train_B358.json'):
            assert (runner.ROOT/name).read_bytes() == bundle.read(name)
    assert len(runner.read(runner.ROOT/'anchors.json')['items']) == 66
    bank = runner.read(runner.ROOT/'frozen_questions.json')['anchor_pool']
    assert len(bank) == 2048
    monkeypatch.syspath_prepend(str(SOURCE))
    spec = importlib.util.spec_from_file_location('real_saved_anchor_validation', SOURCE/'anchor_extension.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'MODEL', '/content/gemma_ngoc_model')
    module.validate_saved(runner.read(runner.ROOT/'anchors.json'), runner.check_sources(), bank[:800])


@pytest.fixture
def installer(runner, monkeypatch):
    import io, subprocess
    source = SOURCE.parent/'install_anchor_extension.py'
    spec = importlib.util.spec_from_file_location('recovery_installer_test', source)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'ROOT', runner.ROOT)
    old = json.loads((Path(__file__).parent/'fixtures/gemma_pool800_plan.json').read_text(encoding='utf-8'))
    for name in ('matched_run.py', 'frozen_questions.json'):
        data = subprocess.check_output(['git', 'show', '729f5a4:src/gemma_ngoc/'+name], cwd=SOURCE.parents[1])
        (runner.ROOT/name).write_bytes(data)
    runner.write(runner.ROOT/'plan.json', old)
    runner.write(runner.ROOT/'experiment_identity.json', {'plan': old, 'software': {}, 'gpu': 'test'})
    for name in ('anchors.json','buckets_train_B440.json','buckets_train_B358.json',
                 'buckets_test_B440.json','results/base.json'):
        runner.write(runner.ROOT/name, {'saved': name})
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: '')
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda url, **kw: io.BytesIO((SOURCE/url.rsplit('/',1)[-1]).read_bytes()))
    return m


def test_installer_updates_only_verified_sources_and_preserves_output(installer):
    m = installer
    saved = (m.ROOT/'anchors.json').read_bytes()
    fixed = m.install('a'*40)
    assert fixed.ROOT == m.ROOT
    assert fixed.check_sources()['anchor_pool'] == 2048
    assert (m.ROOT/'anchors.json').read_bytes() == saved
    assert fixed.read(m.ROOT/'experiment_identity.json')['plan'] == fixed.check_sources()
    assert fixed.read(m.ROOT/'results/anchor_extension/previous_plan.json')['anchor_pool'] == 800


def test_installer_rejects_active_training_and_modified_source(installer, monkeypatch):
    m = installer
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: f'python -u {m.ROOT / "matched_run.py"} prepare')
    with pytest.raises(RuntimeError, match='still active'):
        m.install('a'*40)
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: '')
    (m.ROOT/'common.py').write_text('unexpected edit')
    with pytest.raises(ValueError, match='Existing source was modified'):
        m.install('a'*40)
