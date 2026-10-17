"""Resume anchor selection, extending only the pre-frozen unused candidate bank."""
import json
from pathlib import Path
import statistics

from build_anchors import noise_floors, norm_q, pick_anchors
from common import chat_messages, is_correct
from matched_run import MODEL, check_sources, read, write

ROOT = Path(__file__).resolve().parent


def validate_saved(saved, plan, bank):
    meta, items = saved['meta'], saved['items']
    expected = {'model': str(MODEL), 'dataset': plan['dataset'], 'train_limit': plan['train_count'],
                'k_select': 8, 'k_check': 4, 'hi': .75, 'lo': .25, 'max_len': plan['max_tokens'],
                'temperature': 1.0}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Saved anchors use different model, data, decoding or selection settings.')
    allowed = {norm_q(r['question']): r['answer'] for r in bank}
    seen = set()
    for row in items:
        key = norm_q(row['question'])
        if (key in seen or key not in allowed or row['answer'] != allowed[key]
                or row['p_select'] < .75 or not row.get('base_response')
                or not row.get('base_response_ids')):
            raise ValueError('Saved anchor does not match the frozen eligible candidate bank.')
        seen.add(key)
    if meta.get('n_anchor') != len(items):
        raise ValueError('Saved anchor count differs from its metadata.')


def build(generate):
    plan = check_sources()
    frozen = read(ROOT/'frozen_questions.json')
    bank = frozen['anchor_pool']
    initial, target = plan['anchor_initial_pool'], plan['anchor_count']
    blocked = {norm_q(r['question']) for split in ('train', 'test') for r in frozen[split]}
    keys = [norm_q(r['question']) for r in bank]
    if len(bank) != plan['anchor_pool'] or len(keys) != len(set(keys)) or blocked.intersection(keys):
        raise ValueError('Anchor candidates must be unique and disjoint from frozen train/test data.')
    output = ROOT/'anchors.json'
    if output.exists():
        saved = read(output); validate_saved(saved, plan, bank)
        if len(saved['items']) == target:
            print(f'Anchors already complete: {target}; no generation repeated.', flush=True)
            return
    folder = ROOT/'results/anchor_extension'
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder/'selection.json'
    identity = {'plan': plan, 'environment': read(ROOT/'experiment_identity.json')}
    if cache_path.exists():
        cache = read(cache_path)
        if cache['identity'] != identity:
            raise ValueError('Cached selection belongs to a different experiment or environment.')
    else:
        cache = {'identity': identity, 'retained': [], 'initial_count': 0,
                 'initial_correct_sum': 0.0, 'scored': []}
        if output.exists():
            saved = read(output); meta = saved['meta']
            validate_saved(saved, plan, bank[:initial])
            if meta['pool'] != initial or meta['candidates'] != len(saved['items']):
                raise ValueError('Cannot reconstruct the initial pool from a truncated anchor selection.')
            write(folder/'anchors_initial.json', saved)
            cache.update(retained=saved['items'], initial_count=initial,
                         initial_correct_sum=meta['pool_base_acc'] * initial)
            print(f'Reusing {len(saved["items"])} qualified anchors from {initial} tested candidates.', flush=True)
        write(cache_path, cache)

    def score_until(end):
        start = cache['initial_count'] + len(cache['scored'])
        while start < end:
            batch = bank[start:min(start + plan['anchor_batch_size'], end)]
            outputs = generate(batch, 8, 0)
            if len(outputs) != len(batch) or any(len(o.outputs) != 8 for o in outputs):
                raise ValueError('Incomplete anchor selection batch; prior cached batches are preserved.')
            scored = []
            for row, out in zip(batch, outputs):
                ok = [is_correct(c.text, row['answer']) for c in out.outputs]
                good = next((c for c, correct in zip(out.outputs, ok) if correct), None)
                scored.append({**row, 'p_select': sum(ok)/8, 'base_first_correct': bool(ok[0]),
                               'base_response': good.text if good else None,
                               'base_response_ids': list(good.token_ids) if good else None,
                               'base_response_len': len(good.token_ids) if good else None})
            cache['scored'].extend(scored)
            write(cache_path, cache)
            start += len(batch)
            print(f'[selection saved] {start}/{end} candidates; '
                  f'{sum(r["p_select"] >= .75 for r in cache["retained"] + cache["scored"])} qualified',
                  flush=True)

    score_until(initial)
    eligible = cache['retained'] + cache['scored']
    if cache.get('extended') or sum(r['p_select'] >= .75 for r in eligible) < target:
        cache['extended'] = True
        write(cache_path, cache)
        print(f'Initial pool insufficient; extending to the fixed maximum {len(bank)} candidates.', flush=True)
        score_until(len(bank))
        eligible = cache['retained'] + cache['scored']
    order = {key: i for i, key in enumerate(keys)}
    eligible.sort(key=lambda row: order[norm_q(row['question'])])
    anchors = pick_anchors(eligible, target, .75, 0)
    if len(anchors) != target:
        raise RuntimeError(f'Found {len(anchors)}/{target} eligible anchors after the fixed pool; '
                           'all selection results are cached. No threshold or training count was changed.')
    if 'calibrated' in cache:
        anchors = cache['calibrated']
    else:
        print(f'[calibration] {target} anchors x 4 independent responses', flush=True)
        outputs = generate(anchors, 4, 1)
        if len(outputs) != target or any(len(o.outputs) != 4 for o in outputs):
            raise ValueError('Incomplete calibration; selection cache is preserved.')
        for row, out in zip(anchors, outputs):
            ok = [is_correct(c.text, row['answer']) for c in out.outputs]
            row.update(p_check=sum(ok)/4, check_first_correct=bool(ok[0]))
        cache['calibrated'] = anchors
        write(cache_path, cache)
    tested = cache['initial_count'] + len(cache['scored'])
    meta = {'model': str(MODEL), 'dataset': plan['dataset'], 'train_limit': plan['train_count'],
            'train_total': len(frozen['train']) + len(bank), 'train_used': len(frozen['train']),
            'train_rest': len(bank), 'dup_with_test_removed': 0, 'pool': tested,
            'candidates': sum(r['p_select'] >= .75 for r in eligible), 'n_anchor': target,
            'k_select': 8, 'k_check': 4, 'hi': .75, 'lo': .25, 'max_len': plan['max_tokens'],
            'temperature': 1.0, 'initial_pool': initial, 'pool_max': len(bank),
            'reused_qualified': len(cache['retained']), 'selection_batch_size': plan['anchor_batch_size'],
            'pool_base_acc': (cache['initial_correct_sum'] + sum(r['p_select'] for r in cache['scored']))/tested,
            'anchor_resp_len_mean': statistics.mean(r['base_response_len'] for r in anchors),
            **noise_floors(anchors, .25)}
    result = {'meta': meta, 'items': anchors}
    validate_saved(result, plan, bank)
    write(output, result)
    print(json.dumps(meta, indent=2), flush=True)


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    llm = tokenizer = None

    def generate(rows, n, seed):
        nonlocal llm, tokenizer
        if llm is None:
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
            llm = LLM(str(MODEL), gpu_memory_utilization=.85, max_model_len=1536 + 1024)
        prompts = [tokenizer.apply_chat_template(chat_messages(r['question']), tokenize=False,
                                                 add_generation_prompt=True) for r in rows]
        return llm.generate(prompts, SamplingParams(n=n, temperature=1.0, top_p=1.0,
                                                    max_tokens=1536, seed=seed))

    build(generate)


if __name__ == '__main__':
    import os, sys, traceback
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    except KeyboardInterrupt:
        code = 130
    # The supervisor reaps vLLM workers; do not wait for their Python shutdown hooks.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
