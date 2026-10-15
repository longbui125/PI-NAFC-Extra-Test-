"""All-seed paired evaluation; robust retention and token-cap sensitivity."""
import numpy as np
from io_utils import read, write
from report import question_rates, retention, paired_interval
from engine import PAIRED_ARMS as ARMS, test_path


def report(root):
    shared = root/'shared'
    c = read(shared/'config.json'); data = read(shared/'splits.json'); bundle = read(shared/'prepared.json')
    identity = read(shared/'identity.json')
    seeds = c['seeds']
    domain = c.get('test_domain', 'omni')
    runs, warnings = {}, []
    for seed in seeds:
        folder = root/f'seed_{seed}'
        expected = dict(identity, config=dict(c, seed=seed))
        if not (folder/'identity.json').exists() or read(folder/'identity.json') != expected:
            raise ValueError(f'Seed {seed}: mismatched or missing experiment identity.')
        runs[seed] = {}
        for arm in ARMS:
            path = folder/arm/'complete.json'
            if not path.exists() or read(path)['step'] != c['steps']:
                raise ValueError('All prespecified seeds and arms must finish the planned budget before comparison.')
            runs[seed][arm] = read(path)
            if runs[seed][arm].get('test_steps') != c['test_steps']:
                raise ValueError('Completed run does not match the prespecified test milestones.')
            if runs[seed][arm]['optimizer_updates'] != c['steps']:
                warnings.append(f'Seed {seed}, {arm}: skipped optimizer updates; examine actual update budgets.')
        if runs[seed]['grpo_pi_nafc']['anchor_prompts'] == 0:
            warnings.append(f'Seed {seed}: NAFC never replayed; active protection was not demonstrated.')
    endpoints, trajectory = {}, {}
    primary_initial = None
    for limit in ('primary', 'long'):
        prefix = '' if limit == 'primary' else 'long_'
        questions = data[f'{domain}_test']
        initial = question_rates(read(shared/f'reference_{prefix}{domain}_test.json')['records'], questions, 8)
        noise = question_rates(read(shared/f'reference_{prefix}{domain}_noise.json')['records'], questions, 4)
        stable = initial['accuracy'] >= .75
        if limit == 'primary': primary_initial = initial['accuracy']
        low = initial['accuracy'] <= .25
        noise_retention = retention(initial['accuracy'], noise['accuracy'])
        for step in c['test_steps'] if limit == 'primary' else [c['steps']]:
            endpoint = {'reference': {k: float(v.mean()) for k, v in initial.items()},
                        'unchanged_noise': noise_retention, 'initially_low_success': int(low.sum()),
                        'seeds': {}, 'comparisons': {}}
            raw = {}
            for seed in seeds:
                endpoint['seeds'][str(seed)] = {}
                for arm in ARMS:
                    folder = root/f'seed_{seed}'/arm
                    path = test_path(folder, step, c['steps'], domain) if limit == 'primary' else folder/f'long_{domain}_test.json'
                    if not path.exists():
                        raise ValueError(f'Missing prespecified test milestone: {path}')
                    rr = read(path)['records']
                    current = question_rates(rr, questions, 4)
                    raw[seed, arm] = current
                    f = retention(initial['accuracy'], current['accuracy'])
                    endpoint['seeds'][str(seed)][arm] = {
                        **{k: float(v.mean()) for k, v in current.items()}, 'retention': f,
                        'excess_flip_rate': None if f['rate'] is None else f['rate']-noise_retention['rate'],
                        'newly_reliable': int((low & (current['accuracy'] >= .75)).sum()),
                        'fix_rate': float((current['accuracy'][low] >= .75).mean()) if low.any() else None,
                        'regressed_with_capped_response': int((stable & (current['accuracy'] <= .25) & (current['capped'] > 0)).sum()),
                        'primary_cohort_retention': retention(primary_initial, current['accuracy']),
                        'regressions_by_topic': {topic: retention(initial['accuracy'], current['accuracy'],
                            stable & np.asarray([q['topic'] == topic for q in questions]))
                            for topic in sorted({q['topic'] for q in questions})}}
            for control in ARMS[:-1]:
                accuracy = np.stack([raw[s, 'grpo_pi_nafc']['accuracy']-raw[s, control]['accuracy'] for s in seeds])
                flips = np.stack([(raw[s, 'grpo_pi_nafc']['accuracy'][stable] <= .25).astype(float)-
                                  (raw[s, control]['accuracy'][stable] <= .25).astype(float) for s in seeds])
                def describe(values):
                    if values.shape[1] == 0:
                        return {'mean': None, 'seed_values': [None]*len(seeds), 'seed_sd': None, 'question_ci95': None}
                    per_seed = values.mean(axis=1)
                    return {'mean': float(per_seed.mean()), 'seed_values': per_seed.tolist(),
                            'seed_sd': float(per_seed.std(ddof=1)),
                            'question_ci95': paired_interval(values.mean(axis=0))}
                endpoint['comparisons'][control] = {'accuracy': describe(accuracy), 'flip_rate': describe(flips)}
            if stable.sum() < 128:
                warnings.append(f'{limit} step {step}: only {stable.sum()} initially stable questions; more steps cannot fix a small retention denominator.')
            if noise_retention['rate'] is not None and all(
                    endpoint['seeds'][str(s)]['grpo_pi']['retention']['rate'] <= noise_retention['rate'] for s in seeds):
                warnings.append(f'{limit} step {step}: control forgetting did not exceed unchanged-reference noise in any seed; active protection evidence is limited.')
            endpoint['step'] = step
            if limit == 'primary': trajectory[str(step)] = endpoint
            if step == c['steps']: endpoints[limit] = endpoint
    warnings += [f'{len(seeds)} training seeds provide limited seed-variation evidence; question intervals are conditional on these seeds and one frozen split.',
        data['provenance'].get('scope', 'Fixed benchmark subset; not a full official leaderboard score.'),
        'All capped and unboxed responses remain in primary denominators. Long-cap evaluation is prespecified, not a replacement chosen after results.',
        'Exact deduplication does not certify absence of paraphrases or model pretraining contamination.',
        'All prespecified milestones are reported. Question intervals are pointwise descriptive intervals, not corrected for multiple comparisons.',
        'This paired experiment tests the whole NAFC method; it does not isolate adaptive control from ordinary replay.']
    payload = {'protocol': identity['protocol'], 'model': c['base_model'], 'domain': domain, 'seeds': seeds, 'questions': len(data[f'{domain}_test']),
        'anchors': len(bundle['anchors']), 'allowance': bundle['eps'], 'pi_budget': bundle['budget'],
        'endpoints': endpoints, 'trajectory': trajectory, 'test_steps': c['test_steps'], 'runs': runs, 'warnings': warnings, 'data_provenance': data['provenance']}
    write(root/'comparison.json', payload)
    lines = [f'# {c["base_model"]}: {domain} evaluation', '', f'All prespecified seeds: {seeds}; primary test steps: {c["test_steps"]}. Long-cap sensitivity at the final step only.', '',
        '| Cap | Seed | Arm | Accuracy | Robust regressions | Newly reliable | Capped |',
        '|---|---:|---|---:|---:|---:|---:|']
    reported = [(f'primary step {step}', trajectory[str(step)]) for step in c['test_steps']] + [(f'long step {c["steps"]}', endpoints['long'])]
    for limit, endpoint in reported:
        ref = endpoint['reference']; noise = endpoint['unchanged_noise']
        lines.append(f"| {limit} | — | Reference (8 draws) | {ref['accuracy']:.2%} | unchanged noise {noise['correct_to_low_success']}/{noise['initially_stable']} | — | {ref['capped']:.2%} |")
        for seed in seeds:
            for arm, r in endpoint['seeds'][str(seed)].items():
                f = r['retention']
                lines.append(f"| {limit} | {seed} | {arm} | {r['accuracy']:.2%} | {f['correct_to_low_success']}/{f['initially_stable']} | {r['newly_reliable']}/{endpoint['initially_low_success']} | {r['capped']:.2%} |")
    lines += ['', '| Cap | Comparator | NAFC accuracy delta (pp) | NAFC regression delta (pp) |', '|---|---|---:|---:|']
    for limit, endpoint in reported:
        for control, d in endpoint['comparisons'].items():
            def estimate(v): return 'undefined' if v['mean'] is None else f"{100*v['mean']:+.2f} (seed SD {100*v['seed_sd']:.2f})"
            lines.append(f"| {limit} | {control} | {estimate(d['accuracy'])} | {estimate(d['flip_rate'])} |")
    lines += ['', 'Robust regression: initial success >=6/8, current success <=1/4. Newly reliable: initial <=2/8, current >=3/4.',
        'Negative regression delta and positive accuracy delta favour NAFC. Conditional paired question intervals and all seed values are in comparison.json.',
        'No seed deletion, best-checkpoint selection or pooling with the earlier small pilot.', '', *[f'- {w}' for w in warnings]]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    plot(root, payload)
    print('\n'.join(lines), flush=True)
    return payload


def plot(root, payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    results = payload['endpoints']['primary']['seeds']
    positions = np.arange(len(ARMS))
    labels = ['GRPO + PI', '+ NAFC']
    for ax, key, title in ((axes[0], 'accuracy', 'Sampled accuracy (%)'), (axes[1], 'retention', 'Robust regressions (%)')):
        values = np.asarray([[results[str(s)][a][key]['rate'] if key == 'retention' else results[str(s)][a][key]
                              for a in ARMS] for s in payload['seeds']], dtype=float)*100
        if np.isfinite(values).any():
            ax.bar(positions, values.mean(axis=0), yerr=values.std(axis=0, ddof=1), capsize=4)
            for i, row in enumerate(values): ax.scatter(positions+(i-1)*.07, row, label=f"seed {payload['seeds'][i]}", s=20)
        else:
            ax.text(.5, .5, 'No initially stable test questions', ha='center', transform=ax.transAxes)
        ax.set(xticks=positions, xticklabels=labels, ylabel=title); ax.tick_params(axis='x', rotation=15)
    axes[0].axhline(payload['endpoints']['primary']['reference']['accuracy']*100, color='black', linestyle='--', label='Reference')
    noise = payload['endpoints']['primary']['unchanged_noise']['rate']
    if noise is not None: axes[1].axhline(noise*100, color='black', linestyle='--', label='Unchanged noise')
    for seed in payload['seeds']:
        audits = read(root/f'seed_{seed}'/'grpo_pi_nafc'/'audit_metrics.json')
        axes[2].plot([a['step'] for a in audits], [a['lam'] for a in audits], label=f'seed {seed}')
    axes[2].set(title='NAFC multiplier', xlabel='Training batch', ylabel='Lambda')
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'overview.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    steps = payload['test_steps']
    for arm, label in zip(ARMS, labels):
        for ax, metric, title in zip(axes, ('accuracy', 'retention', 'tokens'),
                                     ('Sampled accuracy (%)', 'Robust regressions (%)', 'Mean response tokens')):
            values = np.asarray([[payload['trajectory'][str(step)]['seeds'][str(seed)][arm][metric]['rate']
                                  if metric == 'retention' else payload['trajectory'][str(step)]['seeds'][str(seed)][arm][metric]
                                  for step in steps] for seed in payload['seeds']], dtype=float)
            if metric != 'tokens': values *= 100
            if np.isfinite(values).any():
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                line, = ax.plot(steps, mean, marker='o', label=label)
                ax.fill_between(steps, mean-sd, mean+sd, color=line.get_color(), alpha=.15)
            ax.set(xlabel='Training batch', ylabel=title, xticks=steps)
    axes[0].axhline(payload['endpoints']['primary']['reference']['accuracy']*100, color='black', linestyle='--', label='Reference')
    if noise is not None: axes[1].axhline(noise*100, color='black', linestyle='--', label='Unchanged noise')
    for ax in axes:
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    fig.suptitle('All prespecified checkpoints; mean and training-seed SD')
    fig.tight_layout(); fig.savefig(root/'trajectory.png', dpi=160); plt.close(fig)
