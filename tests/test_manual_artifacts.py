"""Portable full-output ZIP and checkpoint restoration checks."""
import json
from pathlib import Path
import zipfile
import pytest
import torch
import manual_artifacts as artifacts
import io_utils
import engine
import colab_pilot as pilot
from test_pilot import toy_inputs, toy_run, FakeSampler


def fixture(output, config):
    io_utils.write(output/'shared'/'identity.json', {'config': config, 'protocol': pilot.PROTOCOL})
    io_utils.write(output/'shared'/'prepared.json', {'anchors': ['fixed'], 'budget': 20})
    io_utils.write(output/'shared'/'reference.json', {'records': [1, 2]})


def test_archive_roundtrip_includes_seed_state_and_rejects_conflicts(tmp_path):
    c = {'seeds': [42, 43, 44], 'base_model': 'example'}
    output = tmp_path/'original'; fixture(output, c)
    io_utils.write(output/'seed_42'/'identity.json', {'seed': 42})
    (output/'seed_42'/'resume.pt').write_bytes(b'parameters and optimizer')
    (output/'seed_42'/'unfinished.tmp').write_bytes(b'incomplete')
    (output/'REPORT.md').write_text('Verified report')
    shared = artifacts.pack(output, tmp_path/'shared.zip')
    seed = artifacts.pack(output, tmp_path/'seed42.zip', seed=42)
    restored = tmp_path/'restored'
    with pytest.raises(ValueError, match='matching shared'):
        artifacts.unpack([seed], restored, c)
    assert not list(restored.iterdir())
    artifacts.unpack([seed, shared], restored, c)
    assert (restored/'seed_42'/'resume.pt').read_bytes() == b'parameters and optimizer'
    assert not (restored/'seed_42'/'unfinished.tmp').exists()
    assert (restored/'REPORT.md').read_text() == 'Verified report'
    artifacts.unpack([shared, seed], restored, c)
    (output/'seed_42'/'resume.pt').write_bytes(b'new checkpoint')
    newer = artifacts.pack(output, tmp_path/'newer.zip', seed=42)
    with pytest.raises(ValueError, match='Conflicting'):
        artifacts.unpack([seed, newer, shared], tmp_path/'conflict', c)
    with pytest.raises(ValueError, match='Existing results'):
        artifacts.unpack([newer], restored, c)
    with pytest.raises(ValueError, match='model/configuration'):
        artifacts.unpack([shared], tmp_path/'wrong', dict(c, base_model='different'))


@pytest.mark.parametrize('name', ['../escape.txt', 'shared/../../escape.txt', '/escape.txt',
                                  'C:/escape.txt', 'shared\\escape.txt', 'seed_42/escape.txt'])
def test_untrusted_archive_paths_are_rejected_before_extraction(tmp_path, name):
    c = {'seeds': [42]}
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr(artifacts.MANIFEST, json.dumps({'format': 1, 'identity': {'config': c}, 'seed': None}))
        bundle.writestr(name, 'unsafe')
    if '\\' in name:
        path.write_bytes(path.read_bytes().replace(name.replace('\\', '/').encode(), name.encode()))
    output = tmp_path/'out'
    with pytest.raises(ValueError, match='Unsafe'):
        artifacts.unpack([path], output, c)
    assert not list(output.iterdir()) and not (tmp_path/'escape.txt').exists()


def test_different_anchor_calibrations_cannot_be_pooled(tmp_path):
    c = {'seeds': [42]}
    a = tmp_path/'a'; fixture(a, c)
    io_utils.write(a/'seed_42'/'identity.json', {'seed': 42})
    seed = artifacts.pack(a, tmp_path/'seed.zip', seed=42)
    b = tmp_path/'b'; fixture(b, c)
    io_utils.write(b/'shared'/'prepared.json', {'anchors': ['different'], 'budget': 20})
    shared = artifacts.pack(b, tmp_path/'shared.zip')
    with pytest.raises(ValueError, match='matching shared'):
        artifacts.unpack([shared, seed], tmp_path/'out', c)


def test_full_notebook_zip_preserves_all_seeds_reports_and_excludes_credentials(tmp_path):
    c = {'seeds': [42, 43, 44]}
    output = tmp_path/'output'; fixture(output, c)
    for seed in (42, 43):
        for arm in ('grpo_pi', 'grpo_pi_nafc'):
            io_utils.write(output/f'seed_{seed}'/arm/'resume.pt', {'optimizer': seed, 'scheduler': 3})
    (output/'REPORT.md').write_text('results')
    (output/'credentials.conf').write_text('must not be exported')
    io_utils.write(output/'hf_cache'/'token.json', {'token': 'private'})
    archive = artifacts.pack(output, tmp_path/'notebook.zip', seed='all')
    with zipfile.ZipFile(archive) as bundle:
        assert json.loads(bundle.read(artifacts.MANIFEST))['seed'] == 'all'
        assert not any('credentials' in name or 'hf_cache' in name for name in bundle.namelist())
        assert all(f'seed_{seed}/{arm}/resume.pt' in bundle.namelist()
            for seed in (42, 43) for arm in ('grpo_pi', 'grpo_pi_nafc'))
    restored = tmp_path/'restored'; artifacts.unpack([archive], restored, c)
    assert (restored/'REPORT.md').read_text() == 'results'
    assert io_utils.read(restored/'seed_43'/'grpo_pi_nafc'/'resume.pt')['optimizer'] == 43
    assert not (restored/'seed_44').exists()


def test_partial_preparation_can_be_exported_before_training(tmp_path):
    c = {'seeds': [42, 43, 44]}
    output = tmp_path/'output'; fixture(output, c)
    (output/'shared'/'prepared.json').unlink()
    archive = artifacts.pack(output, tmp_path/'partial.zip', seed='all')
    restored = tmp_path/'restored'; artifacts.unpack([archive], restored, c)
    assert (restored/'shared'/'reference.json').exists()
    assert not (restored/'shared'/'prepared.json').exists()


def test_restoring_seed_zip_preserves_optimizer_and_model_training(tmp_path, monkeypatch):
    c, ds, bundle = toy_inputs()
    c.update(steps=4, checkpoint_every=1, validation_steps=[], test_steps=[2, 4], seeds=[2, 3, 4])
    monkeypatch.setattr(FakeSampler, 'bad_anchors', True)
    monkeypatch.setattr(FakeSampler, 'fail_step', None)
    plain = tmp_path/'plain'; plain.mkdir(); monkeypatch.setattr(engine, 'OUT', plain)
    toy_run(c, ds, bundle, 'grpo_pi_nafc')
    expected = torch.load(plain/'grpo_pi_nafc'/'resume.pt', weights_only=True)
    output = tmp_path/'out'; fixture(output, c)
    root = output/'seed_2'; root.mkdir(); monkeypatch.setattr(engine, 'OUT', root)
    monkeypatch.setattr(FakeSampler, 'fail_step', 2)
    with pytest.raises(RuntimeError): toy_run(c, ds, bundle, 'grpo_pi_nafc')
    archive = artifacts.pack(output, tmp_path/'notebook.zip', seed='all')
    restored = tmp_path/'new_session'; artifacts.unpack([archive], restored, c)
    monkeypatch.setattr(engine, 'OUT', restored/'seed_2')
    monkeypatch.setattr(FakeSampler, 'fail_step', None)
    toy_run(c, ds, bundle, 'grpo_pi_nafc')
    final = torch.load(restored/'seed_2'/'grpo_pi_nafc'/'resume.pt', weights_only=True)
    for name, value in expected['parameters'].items():
        torch.testing.assert_close(final['parameters'][name], value, rtol=0, atol=0)
    assert final['scheduler'] == expected['scheduler']
    assert final['state'] == expected['state'] or all(final['state'][key] == expected['state'][key] for key in ('pi','nafc','replay_ptr','step'))
    for index, state in expected['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(final['optimizer']['state'][index][name], value, rtol=0, atol=0)
