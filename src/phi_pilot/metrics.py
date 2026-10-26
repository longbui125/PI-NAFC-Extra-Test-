"""Aligned final-answer metrics and reports; no model imports."""
import csv
import math
from pathlib import Path
import statistics as stats

from hooks import read, save

LABELS = {'mon':'GRPO + PI', 'rep':'GRPO + PI + NAFC',
          'quota':'GRPO + PI + quota replay', 'kl':'GRPO + PI + KL'}


def summarize(reference, evaluations):
    expected = [r['id'] for r in reference]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Reference IDs must be nonempty and unique.')
    rows = []
    for arm, result in evaluations.items():
        questions = result['per_question']
        if [r['id'] for r in questions] != expected:
            raise ValueError('Evaluation question IDs/order differ: '+arm)
        if any(not math.isfinite(r['acc']) or not 0 <= r['acc'] <= 1 for r in questions):
            raise ValueError('Invalid per-question accuracy: '+arm)
        accuracy = 100*stats.mean(r['acc'] for r in questions)
        length = stats.mean(r['len'] for r in questions)
        if not math.isclose(accuracy,result['overall']['acc'],abs_tol=1e-8):
            raise ValueError('Aggregate accuracy disagrees with per-question scores: '+arm)
        if not math.isclose(length,result['overall']['mean_len'],abs_tol=1e-8):
            raise ValueError('Aggregate length disagrees with per-question scores: '+arm)
        rows.append(dict(arm=arm, accuracy_pct=accuracy,
            robust_regressions=sum(a['p']>=.75 and b['acc']<=.25 for a,b in zip(reference,questions)),
            newly_reliable=sum(a['p']<=.25 and b['acc']>=.75 for a,b in zip(reference,questions)),
            mean_tokens=length))
    return rows


def write_report(folder, rows, title):
    folder = Path(folder)
    with (folder/'summary.csv').open('w',encoding='utf8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = '# '+title+'\n\n'
    text += '| Arm | Accuracy (%) | Robust regressions | Newly reliable | Mean tokens |\n'
    text += '|---|---:|---:|---:|---:|\n'
    for row in rows:
        text += f"| {LABELS[row['arm']]} | {row['accuracy_pct']:.4f} | {row['robust_regressions']} | {row['newly_reliable']} | {row['mean_tokens']:.2f} |\n"
    text += '\nRobust regression: reference ≥6/8 correct, final evaluation ≤1/4 correct. '
    text += 'Newly reliable: reference ≤2/8 correct, final evaluation ≥3/4 correct.\n'
    (folder/'REPORT.md').write_text(text,encoding='utf8')
    print(text,flush=True)


def aggregate(folder, seeds, arms, protocol):
    folder = Path(folder)
    reports = []
    for seed in seeds:
        report = read(folder/f'seed_{seed}'/'summary.json')
        identity = read(folder/f'seed_{seed}'/'identity.json')
        if (report['seed'] != seed or report['protocol'] != protocol
                or identity['seed'] != seed or identity['shared']['protocol'] != protocol
                or [r['arm'] for r in report['rows']] != list(arms)):
            raise ValueError('Incomplete or incompatible seed report: '+str(seed))
        if reports and identity['inputs'] != reports[0][1]['inputs']:
            raise ValueError('Seeds used different prepared inputs.')
        reports.append((report,identity))
    rows = []
    for index, arm in enumerate(arms):
        row = dict(arm=arm)
        for metric in ('accuracy_pct','robust_regressions','newly_reliable','mean_tokens'):
            values = [r['rows'][index][metric] for r,_ in reports]
            row[metric+'_mean'] = stats.mean(values)
            row[metric+'_std'] = stats.stdev(values) if len(values)>1 else 0.
        rows.append(row)
    save(folder/'aggregate.json',dict(seeds=list(seeds),rows=rows,protocol=protocol))
    with (folder/'aggregate.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('All three seeds complete; mean and sample standard deviation:',rows,flush=True)
