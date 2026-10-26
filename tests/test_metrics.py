from pathlib import Path

import pytest

from hooks import read, save
from metrics import summarize, aggregate

ROOT=Path(__file__).resolve().parents[1]
ARMS=('mon','rep','quota','kl')


def test_published_seed_one_results_recompute_exactly():
    folder=ROOT/'results/seed_1'
    rows=summarize(read(folder/'reference.json'),{a:read(folder/f'{a}.json') for a in ARMS})
    expected=[(16.075,40,13),(17.6125,32,21),(14.325,54,8),(15.875,36,11)]
    assert [(r['accuracy_pct'],r['robust_regressions'],r['newly_reliable']) for r in rows]==expected
    assert rows==read(folder/'summary.json')['rows']


def test_misaligned_or_inconsistent_scores_are_rejected():
    reference=[{'id':'a','p':.75},{'id':'b','p':.25}]
    result={'per_question':[{'id':'b','acc':1.,'len':2},{'id':'a','acc':0.,'len':2}],
            'overall':{'acc':50.,'mean_len':2.}}
    with pytest.raises(ValueError,match='IDs/order'):summarize(reference,{'mon':result})
    result['per_question'].reverse()
    result['overall']['acc']=60
    with pytest.raises(ValueError,match='Aggregate accuracy'):summarize(reference,{'mon':result})


def test_aggregate_uses_all_three_seeds_and_refuses_changed_inputs(tmp_path):
    for seed in (1,2,3):
        rows=[dict(arm=a,accuracy_pct=10+seed,robust_regressions=30-seed,
                   newly_reliable=seed,mean_tokens=400+seed) for a in ARMS]
        save(tmp_path/f'seed_{seed}/summary.json',dict(seed=seed,protocol='frozen',rows=rows))
        save(tmp_path/f'seed_{seed}/identity.json',dict(seed=seed,shared={'protocol':'frozen'},inputs={'anchors':'same'}))
    aggregate(tmp_path,[1,2,3],ARMS,'frozen')
    row=read(tmp_path/'aggregate.json')['rows'][0]
    assert row['accuracy_pct_mean']==12 and row['accuracy_pct_std']==1
    path=tmp_path/'seed_3/identity.json'
    identity=read(path);identity['inputs']['anchors']='different';save(path,identity)
    with pytest.raises(ValueError,match='different prepared inputs'):
        aggregate(tmp_path,[1,2,3],ARMS,'frozen')
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path,[1,2,4],ARMS,'frozen')
