"""Preserve preparation, refuse checkpoint mixing, and retain the effective batch."""
import json
from pathlib import Path
import subprocess

import pytest
import torch

from test_gemma_ngoc import SOURCE, load_runner, runner
from test_anchor_extension import installer


@pytest.fixture
def micro4_installer(installer):
    m = installer
    old = json.loads((Path(__file__).parent/'fixtures/gemma_micro4_plan.json').read_text(encoding='utf-8'))
    for name in ('matched_run.py','train.py','anchor_extension.py'):
        data = subprocess.run(['git','show','d189f83:src/gemma_ngoc/'+name], cwd=SOURCE.parents[1],
                              check=True, capture_output=True).stdout
        (m.ROOT/name).write_bytes(data)
    (m.ROOT/'plan.json').write_text(json.dumps(old),encoding='utf-8')
    (m.ROOT/'experiment_identity.json').write_text(json.dumps({'plan':old,'software':{},'gpu':'test'}),encoding='utf-8')
    return m


def test_oom_recovery_reuses_preparation_and_archives_only_unsaved_scratch(micro4_installer):
    m = micro4_installer
    saved = {name:(m.ROOT/name).read_bytes() for name in ('anchors.json','results/base.json','buckets_train_B358.json')}
    folder = m.ROOT/'runs/lacPImonH_s1'; folder.mkdir(parents=True)
    (folder/'events.log').write_text('17 steps, no checkpoint')
    (m.ROOT/'logs').mkdir()
    (m.ROOT/'logs/train_lacPImonH_s1.log').write_text('torch.OutOfMemoryError: CUDA out of memory')
    runner = m.install('a'*40)
    assert runner.check_sources()['micro_batch'] == 1
    assert not folder.exists()
    assert len(list((m.ROOT/'results/failed_runs').glob('*/events.log'))) == 1
    for name,data in saved.items():
        assert (m.ROOT/name).read_bytes() == data
    assert runner.read(m.ROOT/'experiment_identity.json')['plan'] == runner.check_sources()


@pytest.mark.parametrize('name',['checkpoint-25/optimizer.pt','trainer_state.json','adapter_model.safetensors'])
def test_memory_migration_refuses_saved_training_before_any_source_change(micro4_installer, name):
    m = micro4_installer
    source_before = (m.ROOT/'matched_run.py').read_bytes()
    plan_before = (m.ROOT/'plan.json').read_bytes()
    path = m.ROOT/'runs/lacPImonH_s1'/name; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'preserve')
    with pytest.raises(ValueError,match='Saved training state exists'):
        m.install('a'*40)
    assert path.read_bytes() == b'preserve'
    assert (m.ROOT/'matched_run.py').read_bytes() == source_before
    assert (m.ROOT/'plan.json').read_bytes() == plan_before


def test_only_known_memory_plan_is_compatible():
    runner = load_runner(); current = runner.check_sources()
    old = json.loads((Path(__file__).parent/'fixtures/gemma_micro4_plan.json').read_text(encoding='utf-8'))
    assert runner.same_experiment(old,current)
    assert not runner.same_experiment({**old,'learning_rate':2e-5},current)
    assert not runner.same_experiment({**old,'micro_batch':2},current)
    assert current['micro_batch'] == 1
    assert current['prompts_per_step']*current['num_generations'] == 64


def test_accumulation_preserves_effective_batch_gradient_and_penalty_weight():
    torch.manual_seed(0)
    initial = torch.nn.Linear(3,2).double().state_dict()
    x = torch.randn(64,3,dtype=torch.float64)
    y = torch.arange(64)%2
    anchor_x = torch.tensor([[1.,2.,3.]],dtype=torch.float64)
    def gradient(micro):
        model = torch.nn.Linear(3,2).double(); model.load_state_dict(initial)
        accumulation = 64//micro
        for start in range(0,64,micro):
            task = torch.nn.functional.cross_entropy(model(x[start:start+micro]),y[start:start+micro])
            penalty = torch.nn.functional.cross_entropy(model(anchor_x),torch.tensor([1]))
            ((task+.5*penalty)/accumulation).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.allclose(gradient(4),gradient(1),rtol=1e-12,atol=1e-12)
