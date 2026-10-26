import importlib.util

import pytest

import hooks
from build_notebook import ROOT


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('test_runner',ROOT/'phi_pilot/runner.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code=tmp_path/'code'; code.mkdir()
    (code/'probe.py').write_text('SOURCE',encoding='utf8')
    hooks.save(code/'manifest.json',{'probe.py':module.file_hash(code/'probe.py')})
    monkeypatch.setattr(module,'CODE',code)
    monkeypatch.setattr(module,'SHARED',tmp_path/'experiment')
    monkeypatch.setattr(module,'OUT',module.SHARED)
    hooks.save(module.SHARED/'identity.json',dict(protocol=module.digest(module.PLAN),
               sources=hooks.read(code/'manifest.json'),versions=module.PLAN['versions']))
    for name in module.PREPARATION:
        hooks.save(module.SHARED/name, {'fixture':name})
    hooks.save(module.SHARED/'readiness.json',{'ready':True,'pi_budget_tokens':400.86})
    return module


def test_seed_setup_shares_preparation_but_not_trained_state(runtime):
    m=runtime
    m.configure_seed(1)
    hooks.save(m.OUT/'runs/rep/private_state.json',{'seed':1})
    original=(m.OUT/'identity.json').read_bytes()
    m.configure_seed(2)
    assert m.SEED==2 and m.OUT.name=='seed_2'
    assert not (m.OUT/'runs').exists()
    assert (m.SHARED/'seed_1/identity.json').read_bytes()==original
    for name in m.PREPARATION:
        assert m.file_hash(m.OUT/name)==m.file_hash(m.SHARED/name)
    m.configure_seed(2)
    assert m.OUT.name=='seed_2'


def test_changed_preparation_and_unidentified_outputs_are_not_overwritten(runtime):
    m=runtime
    m.configure_seed(1)
    identity=(m.OUT/'identity.json').read_bytes()
    hooks.save(m.SHARED/'anchors.json',{'changed':True})
    with pytest.raises(ValueError,match='inputs changed'):
        m.configure_seed(1)
    assert (m.OUT/'identity.json').read_bytes()==identity
    hooks.save(m.SHARED/'seed_2/unknown.json',{'preserve':True})
    with pytest.raises(ValueError,match='unidentified'):
        m.configure_seed(2)


def test_source_tampering_stops_before_seed_creation(runtime):
    (runtime.CODE/'probe.py').write_text('CHANGED',encoding='utf8')
    with pytest.raises(ValueError,match='source changed'):
        runtime.configure_seed(1)
    assert not (runtime.SHARED/'seed_1').exists()


def test_four_arm_args_keep_batch_pi_and_seed(runtime):
    m=runtime
    for seed in (1,2,3):
        m.configure_seed(seed)
        for arm in m.ARMS:
            args=m.training_args(arm,m.OUT/'runs'/arm)
            value=lambda key:args[args.index(key)+1]
            assert value('--seed')==str(seed)
            assert value('--max_steps')=='300'
            assert int(value('--prompts_per_step'))*int(value('--num_generations'))==64
            assert value('--micro_batch')=='1'
            assert value('--kp')=='4' and value('--eta')=='.05'
            assert value('--lambda_max')=='3'
            assert float(value('--beta'))==(.04 if arm=='kl' else 0.)
            assert value('--flip_mode')==dict(mon='monitor',rep='nafc_replay',quota='replay_fixed',kl='monitor')[arm]


def make_checkpoint(folder,step):
    path=folder/f'checkpoint-{step}'
    for name in hooks.REQUIRED:
        hooks.save(path/name,{'global_step':step})
    hooks.save(path/'COMPLETE.json',{n:(path/n).stat().st_size for n in hooks.REQUIRED})
    return path


def test_incomplete_newer_checkpoint_falls_back_without_deleting_it(tmp_path):
    old=make_checkpoint(tmp_path,25)
    new=make_checkpoint(tmp_path,50)
    (new/'optimizer.pt').write_bytes(b'')
    assert hooks.latest_checkpoint(tmp_path)==old
    assert new.exists()
    (old/'scheduler.pt').unlink()
    with pytest.raises(ValueError,match='No complete checkpoint'):
        hooks.latest_checkpoint(tmp_path)


def test_completed_training_is_skipped_and_quota_requires_same_seed_nafc(runtime):
    m=runtime; m.configure_seed(2)
    make_checkpoint(m.OUT/'runs/mon',300)
    m.train('mon')  # Must return without importing GPU-only training packages.
    with pytest.raises(RuntimeError,match='complete NAFC trace'):
        m.train('quota')


def test_quota_validates_matching_step_and_generation_groups(tmp_path):
    hooks.save(tmp_path/'observations/0007.json',dict(step=7,groups=8,replay_groups=3))
    assert hooks.quota(tmp_path,7,8)==3
    with pytest.raises(ValueError):hooks.quota(tmp_path,7,4)
