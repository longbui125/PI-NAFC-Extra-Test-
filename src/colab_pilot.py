"""Paired multi-seed Olympic evaluation, resumable across Colab sessions."""
import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "hf_cache"))
import torch
import engine
import io_utils
import model as modeling
from core import robust_flags, split_anchors
from scoring import math_gold
from data import DATASETS, rows, normalized, take
from io_utils import ROOT, read, write, freeze, digest, file_hash, checkpoint_identity
from model import Sampler, load_model, base_path, prompt_ids, token_logps, evaluate, clear_memory

ARMS = engine.PAIRED_ARMS
PROTOCOL = 'olympic-a100-manual-v6'
MODELS = {
    'microsoft/Phi-4-mini-instruct': 'cfbefacb99257ffa30c83adab238a50856ac3083',
    'google/gemma-3-1b-it': 'dcc83ea841ab6100d6b47a070329e1ba4cf78752'}
OUT = ROOT / "outputs_colab"


def configure(output):
    global OUT
    OUT = Path(output).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    engine.OUT = OUT
    return OUT


def validate(c):
    if c['base_model'] not in MODELS or c['base_revision'] != MODELS[c['base_model']] or c['precision'] != 'fp16':
        raise ValueError('Use the pinned Phi/Gemma model revision in fp16.')
    for k in ("steps", "prompts_per_step", "group_size", "train_count", "validation_count", "test_count",
              "anchor_pool", "anchor_count", "anchor_minimum", "generation_batch", "checkpoint_every"):
        if not isinstance(c[k], int) or c[k] <= 0:
            raise ValueError(f"Invalid positive count: {k}")
    if c["initial_draws"] != 8 or c["eval_draws"] != 4 or c["group_size"] < 2:
        raise ValueError("Preserve the eight-draw selection and four-draw NAFC audit.")
    if c["anchor_minimum"] < 4 or c["anchor_minimum"] % 2 or c["anchor_count"] % 2:
        raise ValueError("Repair and controller halves must each have at least two anchors.")
    if not c["anchor_minimum"] <= c["anchor_count"] <= c["anchor_pool"]:
        raise ValueError("Invalid declared anchor size range")
    if c["test_draws"] != 4 or not c["test_do_sample"] or not c["long_evaluation"]:
        raise ValueError("Preserve four-draw stochastic evaluation and long-cap sensitivity.")
    if c['long_eval_tokens'] <= c['max_new_tokens']:
        raise ValueError('The secondary evaluation must use a larger token cap.')
    if c['max_prompt_tokens'] + c['long_eval_tokens'] > c.get('context_length', 131072):
        raise ValueError('Prompt plus long response exceeds the model context; no silent truncation.')
    if len(c['seeds']) < 3 or len(set(c['seeds'])) != len(c['seeds']) or c['seed'] not in c['seeds']:
        raise ValueError('Use all prespecified independent training seeds.')
    if not 0 < c['replay_frac_max'] < 1:
        raise ValueError('Invalid maximum NAFC group replacement fraction.')
    milestones = c['test_steps']
    if not milestones or milestones != sorted(set(milestones)) or milestones[-1] != c['steps'] or any(
            not isinstance(s, int) or s <= 0 or s > c['steps'] for s in milestones):
        raise ValueError('Test milestones must be ordered, unique, within the training budget and include its final step.')
    if c['anchor_minimum'] != c['anchor_count']:
        raise ValueError('This evaluation requires the full declared anchor set; no smaller denominator.')
    if c['train_count'] < c['prompts_per_step'] or c['train_count'] % c['prompts_per_step']:
        raise ValueError('Training set must form complete prompt groups; epochs repeat with a seeded shuffle.')
    if file_hash(ROOT/"core.py") != read(ROOT/"SOURCE.json")["core_sha256"]:
        raise ValueError("Upstream PI/NAFC core changed")


def hardware(c):
    modeling.check_gpu(c)
    memory = torch.cuda.get_device_properties(0).total_memory/(1024**3)
    if c.get('target_gpu') == 'A100' and ('A100' not in torch.cuda.get_device_name(0) or memory < 35):
        raise RuntimeError('Select an A100 runtime for these notebooks; no automatic hardware downgrade.')
    if memory < 14:
        raise RuntimeError("This fp16 profile requires a GPU with at least 14 GiB usable VRAM, such as Colab T4.")
    print(f"[GPU] {torch.cuda.get_device_name(0)}: {memory:.1f} GiB; fp16; chunked output-head loss", flush=True)


def prepare_data(c):
    frozen = ROOT/c.get('data_file', 'colab_splits.json')
    if frozen.exists():
        saved = read(frozen)
        settings = data_settings(c)
        if saved['settings'] != settings:
            raise ValueError('Frozen notebook data does not match its declared split configuration.')
        data = saved['splits']
        validate_data(data, c)
        print('[data] Using embedded frozen questions; no full-corpus download needed.', flush=True)
        return data
    return build_data(c)


def build_data(c):
    from datasets import load_dataset
    if c.get('test_domain') == 'math':
        import hashlib
        name = 'DigitalLearningGmbH/MATH-lighteval'
        original = {s: load_dataset(name, revision=c['math_revision'], split=s) for s in ('train', 'test')}
        pools, invalid_labels = {}, {}
        for split, dataset in original.items():
            unique = {}; invalid_labels[split] = []
            for i, item in enumerate(dataset):
                key = normalized(item['problem'])
                answer = math_gold(item['solution'])
                if not key:
                    raise ValueError('MATH contains an empty question.')
                if not answer:
                    invalid_labels[split].append(i)
                    continue
                unique.setdefault(key, {'id': hashlib.sha256(f'math/{split}/{key}'.encode()).hexdigest(),
                    'domain': 'math', 'question': item['problem'], 'answer': answer, 'topic': item['type'],
                    'source': name, 'source_index': i, 'difficulty': item.get('level')})
            pools[split] = list(unique.values())
        blocked = {normalized(q['question']) for q in pools['test']}
        pool = [q for q in pools['train'] if normalized(q['question']) not in blocked]
        a, b = c['train_count'], c['train_count']+c['validation_count']
        selected = take(pool, b+c['anchor_pool'], c['split_seed'])
        result = {'task_train': selected[:a], 'math_validation': selected[a:b], 'anchor_candidates': selected[b:],
            'math_test': take(pools['test'], c['test_count'], c['split_seed']+1),
            'provenance': {'task_domain': 'math', 'evaluation_domains': ['math'],
                'datasets': {'math': name}, 'revisions': {'math': c['math_revision']},
                'source_counts': {k: len(v) for k,v in original.items()},
                'excluded_missing_or_empty_gold': invalid_labels,
                'scope': 'Fixed random MATH test subset; all seven topics and answer types, no integer/difficulty/success filter.',
                'deduplication': 'Normalized exact train/test overlap removed before selection.'}}
        validate_data(result, c)
        return result
    external = load_dataset(DATASETS['omni'], revision=c['omni_revision'], split='test')
    blocked = {normalized(q['problem']) for q in external}
    eligible_test = rows(external, 'omni', 'test')
    raw = load_dataset(DATASETS['numina'], revision=c['numina_revision'], split='train')
    pool = [q for q in rows(raw, 'numina', 'research_pool', c['numina_sources'])
            if normalized(q['question']) not in blocked]
    picked = take(pool, c['train_count']+c['validation_count']+c['anchor_pool'], c['split_seed'])
    a = c['train_count']; b = a+c['validation_count']
    data = {'task_train': picked[:a], 'numina_validation': picked[a:b], 'anchor_candidates': picked[b:],
        'omni_test': take(eligible_test, c['test_count'], c['split_seed']+1),
        'provenance': {'task_domain': 'numina', 'evaluation_domains': ['omni'],
            'datasets': DATASETS, 'numina_sources': c['numina_sources'],
            'revisions': {k: c[f'{k}_revision'] for k in DATASETS},
            'eligible_numina': len(pool), 'eligible_omni': len(eligible_test), 'omni_total': len(external),
            'scope': 'Fixed random numeric-answer subset of Omni-MATH; not an official full-benchmark score.',
            'deduplication': 'Normalized exact overlap with ALL Omni questions removed; paraphrases not certified.'}}
    validate_data(data, c)
    return data


def data_settings(c):
    keys = ['split_seed', 'train_count', 'validation_count', 'anchor_pool', 'test_count']
    keys += ['math_revision'] if c.get('test_domain') == 'math' else ['numina_revision', 'omni_revision', 'numina_sources']
    return {k: c[k] for k in keys}


def validate_data(data, c):
    domain = c.get('test_domain', 'omni')
    validation = 'math_validation' if domain == 'math' else 'numina_validation'
    for name, count in (('task_train', c['train_count']), (validation, c['validation_count']),
                        ('anchor_candidates', c['anchor_pool']), (f'{domain}_test', c['test_count'])):
        if len(data[name]) != count or any(not q['answer'] for q in data[name]):
            raise ValueError(f'Invalid frozen question count or answer in {name}')
    all_rows = [q for name, rr in data.items() if name != 'provenance' for q in rr]
    keys = [normalized(q['question']) for q in all_rows]
    ids = [q['id'] for q in all_rows]
    if len(keys) != len(set(keys)) or len(ids) != len(set(ids)):
        raise ValueError("Question leakage between pilot partitions")


def identity(c, data, reference):
    return {'protocol': PROTOCOL, 'config': c, 'splits': digest(data),
        'reference': reference, 'code': {name: file_hash(ROOT/name) for name in
        ('core.py', 'data.py', 'engine.py', 'model.py', 'io_utils.py', 'scoring.py', 'colab_pilot.py', 'report.py', 'colab_report.py', 'manual_artifacts.py')},
        'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'peft', 'math-verify', 'accelerate', 'numpy', 'matplotlib')},
        'gpu': torch.cuda.get_device_name(0)}


def dry_backward(model, tok, c):
    # Validate the worst configured response length before expensive calibration, with zero optimizer updates.
    model.train(); model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    ids = prompt_ids(tok, 'Calculate 2 + 2.')
    token = tok.encode('1', add_special_tokens=False)[-1]
    logps = token_logps(model, ids, [token]*c['max_new_tokens'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c['learning_rate'])
    scaler = torch.cuda.amp.GradScaler(init_scale=c['loss_scale'])
    scaler.scale(-logps.mean()).backward(); scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite preflight gradients; no policy was updated.")
    print(f"[preflight] {c['max_new_tokens']}-token backward peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)
    model.zero_grad(set_to_none=True)
    del optimizer, scaler, logps


def select_anchors(c, data, sampler, shared):
    path = shared/'anchor_selection.json'
    state = read(path) if path.exists() else {'tested': 0, 'records': []}
    pool = data['anchor_candidates']; n = 8
    if [(r['id'], r['sample_index']) for r in state['records']] != [
        (q['id'], j) for q in pool[:state['tested']] for j in range(n)]:
        raise ValueError("Incomplete or reordered anchor-selection progress")
    def stable():
        return [dict(q, base_first_correct=state['records'][i*n]['correct'])
            for i, q in enumerate(pool[:state['tested']])
            if sum(r['correct'] for r in state['records'][i*n:(i+1)*n]) >= 6]
    anchors = stable()
    while len(anchors) < c['anchor_count'] and state['tested'] < len(pool):
        start = state['tested']; batch = pool[start:start+c['anchor_selection_batch']]
        rr = sampler(batch, n, 300_000_000+start*n*1009)
        if [(r['id'], r['sample_index']) for r in rr] != [(q['id'], j) for q in batch for j in range(n)]:
            raise ValueError("Incomplete anchor generation")
        state['records'].extend(rr); state['tested'] += len(batch)
        anchors = stable()
        write(path, state)
        print(f"[anchors] stable={len(anchors)}, target={c['anchor_count']}, tested={state['tested']}/{len(pool)}", flush=True)
    return anchors[:c['anchor_count']] if len(anchors) >= c['anchor_count'] else []


def prepare(c):
    hardware(c)
    shared = OUT/'shared'; shared.mkdir(exist_ok=True)
    freeze(shared/'config.json', c)
    data = read(shared/'splits.json') if (shared/'splits.json').exists() else prepare_data(c)
    freeze(shared/'splits.json', data)
    path = base_path(c)
    freeze(shared/'identity.json', identity(c, data, checkpoint_identity(path)))
    if (shared/'prepared.json').exists():
        print('[prepare] Reusing complete preparation.', flush=True)
        return read(shared/'prepared.json')
    model, tok = load_model(c, path, trainable=True)
    for name, rr in data.items():
        if name != 'provenance' and any(len(prompt_ids(tok, q['question'])) > c['max_prompt_tokens'] for q in rr):
            raise ValueError('A prompt exceeds the cap; no silent truncation.')
    dry_backward(model, tok, c)
    sampler = Sampler(model, tok, c)
    validation = 'math_validation' if c.get('test_domain') == 'math' else 'numina_validation'
    val = evaluate(sampler, data[validation], 4, 410_000_000, shared/'reference_validation.json')
    mean = sum(len(r['tokens']) for r in val)/len(val)
    anchors = select_anchors(c, data, sampler, shared)
    if not anchors:
        diagnostic = {'status': 'insufficient_stable_anchors', 'minimum': c['anchor_minimum'],
            'note': 'Frozen Olympic pool did not supply enough stable anchors. No arm was trained; no efficacy comparison is valid.'}
        write(shared/'diagnostic.json', diagnostic)
        print(diagnostic, flush=True)
        del sampler, model, tok; clear_memory()
        return diagnostic
    repair, held = split_anchors(len(anchors), .5, 0)
    noise = []
    for repeat in range(3):
        rr = evaluate(sampler, [anchors[i] for i in held], 4, 400_000_000+repeat*1_000_000,
                      shared/f'anchor_noise_{repeat}.json')
        noise.append(sum(robust_flags([[r['correct'] for r in rr[i*4:(i+1)*4]] for i in range(len(held))]))/len(held))
    domain = c.get('test_domain', 'omni')
    for prefix, cap in (('', c['max_new_tokens']), ('long_', c['long_eval_tokens'])):
        test_sampler = Sampler(model, tok, dict(c, max_new_tokens=cap, do_sample=True))
        initial = evaluate(test_sampler, data[f'{domain}_test'], 8, 420_000_000 if not prefix else 450_000_000,
                          shared/f'reference_{prefix}{domain}_test.json')
        evaluate(test_sampler, data[f'{domain}_test'], 4, 430_000_000 if not prefix else 440_000_000,
                 shared/f'reference_{prefix}{domain}_noise.json')
        stable = sum(sum(r['correct'] for r in initial[i*8:(i+1)*8]) >= 6 for i in range(len(data[f'{domain}_test'])))
        print(f'[reference {cap}] initially stable={stable}/{len(data[f"{domain}_test"])}', flush=True)
        if stable < 128:
            print('[warning] Fewer than 128 stable test questions: retention precision remains limited. No test rows are removed.', flush=True)
    bundle = {'status': 'prepared', 'anchors': anchors, 'repair_idx': repair, 'held_idx': held,
        'noise_repeats': noise, 'noise_robust': sum(noise)/3, 'eps': sum(noise)/3+c['noise_margin'],
        'budget': max(1, round(mean*c['compression_ratio'])), 'calibration_mean_tokens': mean,
        'anchor_target': c['anchor_count'], 'anchor_actual': len(anchors),
        'identity_sha256': digest(read(shared/'identity.json'))}
    write(shared/'prepared.json', bundle)
    print(f"[prepare] actual anchors={len(anchors)}, B={bundle['budget']}, noise={bundle['noise_robust']:.2%}, epsilon={bundle['eps']:.2%}", flush=True)
    del test_sampler, sampler, model, tok; clear_memory()
    return bundle


def train(c, arm, seed):
    hardware(c)
    shared = OUT/'shared'
    data, bundle = read(shared/'splits.json'), read(shared/'prepared.json')
    path = base_path(c)
    if digest(identity(c, data, checkpoint_identity(path))) != bundle['identity_sha256']:
        raise ValueError('Prepared model, code, environment or configuration changed; refusing mismatched resume.')
    seeded = dict(c, seed=seed)
    folder = OUT/f'seed_{seed}'; folder.mkdir(exist_ok=True)
    freeze(folder/'identity.json', identity(seeded, data, checkpoint_identity(path)))
    engine.OUT = folder
    engine.train_arm(dict(seeded, reference_checkpoint=path), data, bundle, arm)


def report():
    from colab_report import report as summarize
    return summarize(OUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train', 'report', 'all'])
    parser.add_argument('--output', default=str(OUT))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--config', default=str(ROOT/'config_colab.json'))
    args = parser.parse_args()
    c = read(args.config); validate(c); configure(args.output)
    if args.command == 'prepare': prepare(c)
    elif args.command == 'train':
        if not args.arm or args.seed not in c['seeds']:
            parser.error('train requires --arm and a prespecified --seed')
        train(c, args.arm, args.seed)
    elif args.command == 'report': report()
    else:
        import subprocess
        base = [sys.executable, str(ROOT/'colab_pilot.py'), '--config', args.config]
        subprocess.run(base+['prepare', '--output', str(OUT)], check=True)
        if not (OUT/'shared'/'prepared.json').exists(): return
        for seed in c['seeds']:
            for arm in ARMS:
                subprocess.run(base+['train', '--arm', arm, '--seed', str(seed), '--output', str(OUT)], check=True)
        report()


if __name__ == '__main__': main()
