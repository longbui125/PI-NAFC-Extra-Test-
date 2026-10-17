"""Run Ngoc's unchanged PI scripts with Gemma; keep caches outside exported output."""
import argparse
import codecs
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent/'gemma_ngoc_model'
ARMS = ('mon', 'rep', 'fix', 'kl', 'fg')
EXPORT_DIRS = ('runs', 'results', 'logs')
MANIFEST = 'MATCHED_RUN.json'
LEGACY_RUNNER_SHA256 = '07c9d339796b7a7302c806bd79678fc8755a5069e4c8822dd1bb8f5126d21261'
SILENCE_LIMIT = 15 * 60
HEARTBEAT_INTERVAL = 60
POLL_INTERVAL = 0.25


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def check_sources():
    plan = read(ROOT/'plan.json')
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Upstream source changed: '+name)
    return plan


def same_experiment(saved, current):
    legacy = {**current, 'source_sha256': {**current['source_sha256'],
                                         'matched_run.py': LEGACY_RUNNER_SHA256}}
    return saved == current or saved == legacy


def run_name(arm, seed):
    return f'lacPI{arm}H_s{seed}'


def train_args(arm, seed):
    p = read(ROOT/'plan.json')
    if arm not in ARMS or seed not in p['seeds']:
        raise ValueError('Undeclared arm or seed.')
    name = run_name(arm, seed)
    return ['--method', 'laconic', '--buckets', 'buckets_train_B358.json', '--model', str(MODEL),
            '--eta', str(p['pi_eta']), '--lambda_max', str(p['pi_lambda_max']), '--adv_norm', 'decoupled',
            '--two_sided', '--kp', str(p['pi_kp']), '--max_steps', str(p['steps']),
            '--num_generations', str(p['num_generations']), '--prompts_per_step', str(p['prompts_per_step']),
            '--micro_batch', str(p['micro_batch']), '--max_completion_length', str(p['max_tokens']),
            '--lr', str(p['learning_rate']), '--lora_r', str(p['lora_r']), '--vllm_mem', str(p['vllm_mem']),
            '--save_steps', str(p['save_steps']), '--save_total_limit', str(p['save_total_limit']),
            '--keep_steps', ','.join(map(str, p['keep_steps'])), '--anchor_split', str(p['anchor_split']),
            '--split_seed', str(p['split_seed']), '--flip_mode', p['arms'][arm], '--anchors', 'anchors.json',
            '--flip_every', str(p['audit_every']), '--flip_k', str(p['anchor_check_draws']),
            '--flip_eta', str(p['nafc_eta'] if arm == 'rep' else 1.0),
            '--flip_gamma', str(p['fixed_replay_gamma'] if arm == 'fix' else p['flipguard_gamma']),
            '--flip_margin', str(p['noise_margin']), '--flip_lam_max', '1.0', '--flip_per_micro', '1',
            '--replay_frac_max', str(p['replay_frac_max']), '--beta', str(p['beta'] if arm == 'kl' else 0.0),
            '--output_dir', f'runs/{name}', '--seed', str(seed), '--merged_dir', str(ROOT.parent/'merged'/name)]


def descendant_sessions(parent):
    rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
        ['ps', '-eo', 'pid=,ppid=,pgid=,sid='], text=True).splitlines() if line.strip()]
    descendants = {parent}
    while children := {pid for pid, ppid, group, session in rows
                       if ppid in descendants and pid not in descendants}:
        descendants.update(children)
    return {pid for pid, ppid, group, session in rows if pid in descendants and pid == group == session}


def run_logged(command, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = last_output = last_heartbeat = time.monotonic()
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    # A vLLM orphan can hold a pipe open after its parent exits; regular files have no EOF wait.
    with log.open('ab', buffering=0) as stream, log.open('rb') as reader:
        reader.seek(0, os.SEEK_END)
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        print(f'[start] {log.name}: pid={process.pid}', flush=True)
        try:
            while True:
                chunk = reader.read(65536)
                now = time.monotonic()
                if chunk:
                    print(decoder.decode(chunk), end='', flush=True)
                    last_output = now
                if process.poll() is not None:
                    break
                if now - last_output >= SILENCE_LIMIT:
                    raise TimeoutError(f'No new output for {SILENCE_LIMIT:g}s; stopped {log.name}. '
                                       f'Completed output/checkpoints are preserved. Inspect {log}.')
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    print(f'[running] {log.name}: {now-started:.0f}s elapsed; '
                          f'{now-last_output:.0f}s since output; pid={process.pid}', flush=True)
                    last_heartbeat = now
                time.sleep(POLL_INTERVAL)
        finally:
            # vLLM workers belong to this child session; never kill unrelated notebook processes.
            groups = {process.pid}
            try:
                if process.poll() is None and os.name == 'posix':
                    groups.update(descendant_sessions(process.pid))
            finally:
                for group in sorted(groups, reverse=True):
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=10)
            remaining = os.fstat(reader.fileno()).st_size - reader.tell()
            while remaining > 0 and (chunk := reader.read(min(65536, remaining))):
                print(decoder.decode(chunk), end='', flush=True)
                remaining -= len(chunk)
            print(decoder.decode(b'', final=True), end='', flush=True)
        if process.returncode:
            raise RuntimeError(f'Command failed ({process.returncode}); see {log}.')
        print(f'[done] {log.name}: {time.monotonic()-started:.1f}s', flush=True)


def execute(script, args, log):
    entry = [str(ROOT/'run_upstream.py'), script] if script != 'matched_run.py' else [str(ROOT/script)]
    run_logged([sys.executable, '-u', *entry, *map(str, args)], ROOT/'logs'/log)


def check_process_cleanup():
    with tempfile.TemporaryDirectory(prefix='nafc_process_check_') as folder:
        folder = Path(folder)
        code = ('import os,subprocess,sys; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                'print(p.pid,flush=True); os._exit(0)')
        started = time.monotonic()
        run_logged([sys.executable, '-u', '-c', code], folder/'orphan.log')
        if time.monotonic() - started >= 10:
            raise RuntimeError('Process cleanup self-check took too long; do not start GPU work.')
        worker = int((folder/'orphan.log').read_text().strip())
        status = Path(f'/proc/{worker}/stat')
        deadline = time.monotonic() + 2
        while status.exists():
            try:
                state = status.read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state in ('Z', 'X'):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Worker survived process cleanup; do not start GPU work.')
            time.sleep(.05)
    print('[self-check] Orphan holding stdout was cleaned up without waiting for EOF.', flush=True)


def initialize():
    p = check_sources()
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Select an A100 GPU runtime.')
    for package, expected in p['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'{package} must be {expected}; use the notebook setup cell.')
    versions = {name: importlib.metadata.version(name) for name in (*p['versions'], 'math-verify', 'datasets')}
    current = {'plan': p, 'software': versions, 'gpu': torch.cuda.get_device_name(0)}
    target = ROOT/'experiment_identity.json'
    if target.exists():
        saved = read(target)
        if not same_experiment(saved.get('plan'), p) or {**saved, 'plan': p} != current:
            raise ValueError('Saved output belongs to a different model, configuration or environment.')
    write(target, current)
    from huggingface_hub import snapshot_download
    snapshot_download(p['model'], revision=p['model_revision'], local_dir=MODEL,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
    return p


def prepare():
    p = initialize()
    from run_upstream import load_split
    if any(len(load_split(p['dataset'], split, 2000)) != 2000 for split in ('train', 'test')):
        raise ValueError('The frozen Olympic partitions must contain 2,000 questions each.')
    common = ['--model', MODEL, '--dataset', p['dataset'], '--max_len', p['max_tokens'], '--n', 8, '--limit', 2000]
    if not (ROOT/'buckets_train_B440.json').exists():
        execute('prepare_buckets.py', common+['--split', 'train', '--budget', 440,
                '--out', 'buckets_train_B440.json'], 'prepare_train.log')
    if not (ROOT/'buckets_train_B358.json').exists():
        execute('prepare_buckets.py', ['--reuse_samples', 'buckets_train_B440.json', '--budget', 358,
                '--out', 'buckets_train_B358.json'], 'reuse_train_B358.log')
    if not (ROOT/'buckets_test_B440.json').exists():
        execute('prepare_buckets.py', common+['--split', 'test', '--budgets_from', 'buckets_train_B440.json',
                '--out', 'buckets_test_B440.json'], 'prepare_test.log')
    (ROOT/'results/texts').mkdir(parents=True, exist_ok=True)
    if not (ROOT/'results/base.json').exists():
        evaluate_model(MODEL, 'base')
    if not (ROOT/'anchors.json').exists():
        execute('build_anchors.py', ['--model', MODEL, '--dataset', p['dataset'], '--train_limit', 2000,
                '--pool', p['anchor_pool'], '--n_anchor', p['anchor_count'], '--max_len', p['max_tokens'],
                '--k_select', 8, '--k_check', 4, '--seed', 0, '--out', 'anchors.json'], 'anchors.log')
    for name in ('buckets_train_B440.json', 'buckets_train_B358.json', 'buckets_test_B440.json'):
        d = read(ROOT/name)
        if (len(d['items']) != 2000 or d['meta']['max_len'] != 1536
            or d['meta']['model'] != str(MODEL) or d['meta']['dataset'] != p['dataset']):
            raise ValueError('Wrong matched data partition: '+name)
    if len(read(ROOT/'anchors.json')['items']) != 128:
        raise ValueError('Fewer than 128 stable anchors; no arm may train. Preserve the diagnostic output.')
    print('Preparation complete: one 1536-token cap; 128 anchors; five arms x three seeds.', flush=True)


def evaluate_model(path, name):
    p = read(ROOT/'plan.json')
    execute('evaluate.py', ['--model', path, '--buckets', 'buckets_test_B440.json', '--n', p['test_draws'],
            '--seed', p['test_seed'], '--max_len', p['max_tokens'], '--gpu_mem', p['eval_vllm_mem'],
            '--out', f'results/{name}.json', '--save_texts', f'results/texts/{name}.jsonl.gz'], f'eval_{name}.log')


def latest_checkpoint(folder):
    checkpoints = sorted(folder.glob('checkpoint-*'), key=lambda path: int(path.name.split('-')[-1]), reverse=True)
    for path in checkpoints:
        needed = ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json', 'optimizer.pt',
                  'scheduler.pt', 'rng_state.pth', 'lambda_state.json', 'flip_state.json')
        if all((path/name).exists() for name in needed):
            return path
    if checkpoints:
        raise ValueError('No complete resumable checkpoint; keep the output and inspect it before restarting.')
    return None


def merge_adapter(checkpoint, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cpu')
    model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(output)


def train_one(arm, seed):
    initialize()
    if not (ROOT/'anchors.json').exists() or len(read(ROOT/'anchors.json')['items']) != 128:
        raise ValueError('Complete preparation before training.')
    name = run_name(arm, seed); folder = ROOT/'runs'/name
    args = train_args(arm, seed)
    if not (folder/'run_info.json').exists():
        checkpoint = latest_checkpoint(folder)
        if checkpoint:
            args += ['--resume', str(checkpoint)]
        elif folder.exists() and any(folder.iterdir()):
            raise ValueError('Partial run without a complete checkpoint; refusing to silently restart.')
        execute('train.py', args, f'train_{name}.log')
    p = read(ROOT/'plan.json')
    for step in [p['steps'], *p['keep_steps']]:
        key = name if step == p['steps'] else f'{name}_step{step}'
        if (ROOT/f'results/{key}.json').exists():
            continue
        checkpoint = folder/(f'checkpoint-{step}' if step == p['steps'] else f'keep/step-{step}')
        merged = ROOT.parent/'merged'/key
        # Merge in a separate process, releasing CPU tensors before the vLLM evaluator starts.
        if not (merged/'config.json').exists():
            execute('matched_run.py', ['merge', '--checkpoint', checkpoint, '--merged', merged], f'merge_{key}.log')
        evaluate_model(merged, key)
        if not merged.resolve().is_relative_to((ROOT.parent/'merged').resolve()):
            raise ValueError('Unexpected merged-model cleanup path.')
        shutil.rmtree(merged)


def report():
    from fixflip import Scores, fixflip, compare
    p = check_sources()
    base = Scores.load(str(ROOT/'buckets_test_B440.json'))
    noise = fixflip(base, Scores.load(str(ROOT/'results/base.json')))
    table = ['# Gemma / Olympic mathematics: Ngoc-matched PI configuration', '',
             '| Step | Seed | Arm | Accuracy (%) | Mean tokens | Robust regressions | Newly reliable | Boxed (%) | Capped (%) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    payload = {'plan': p, 'noise': noise, 'milestones': {}}
    for step in p['test_steps']:
        paired = {}
        payload['milestones'][str(step)] = {}
        for seed in p['seeds']:
            paired[seed] = {}
            for arm in ARMS:
                name = run_name(arm, seed)+(f'_step{step}' if step != 300 else '')
                path = ROOT/f'results/{name}.json'
                if not path.exists():
                    raise ValueError('All prespecified seeds, arms and milestones are required: '+name)
                d = read(path); score = Scores.load(str(path)); paired[seed][arm] = score
                metrics = fixflip(base, score)
                payload['milestones'][str(step)][name] = metrics
                o = d['overall']
                broken = int(((base.p >= .75) & (score.p <= .25)).sum())
                fixed = int(((base.p <= .25) & (score.p >= .75)).sum())
                table.append(f"| {step} | {seed} | {arm} | {o['acc']:.2f} | {o['mean_len']:.1f} | {broken} | {fixed} | {o['boxed_rate']:.2f} | {o['truncated_rate']:.2f} |")
        comparisons = {}
        for control in ('mon', 'fix', 'kl', 'fg'):
            comparisons[control] = {str(seed): compare(base, paired[seed]['rep'], paired[seed][control])
                                    for seed in p['seeds']}
        payload['milestones'][str(step)]['paired_comparisons'] = comparisons
    table += ['', f"Base sampled accuracy: {100*base.acc:.2f}%. Unchanged-reference robust flip rate: {100*noise['robust_flip']:.2f}%.",
              '', 'All five arms and all three seeds are reported. Question-bootstrap intervals in comparison.json are conditional on each seed.',
              'Qwen-specific between-run variance estimates and numerical success thresholds are not transferred to Gemma.',
              'Historical 2048-token HF output is excluded. No longer-cap evaluation or best-checkpoint selection.']
    write(ROOT/'comparison.json', payload)
    (ROOT/'REPORT.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    print('\n'.join(table), flush=True)


def pack(archive):
    p = check_sources()
    allowed = {'plan.json', 'experiment_identity.json', 'anchors.json', 'buckets_train_B440.json',
               'buckets_train_B358.json', 'buckets_test_B440.json', 'comparison.json', 'REPORT.md'}
    archive = Path(archive).resolve()
    if archive.is_relative_to(ROOT):
        raise ValueError('Save the archive outside the experiment folder.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(archive)+'.tmp')
    files = [path for path in ROOT.rglob('*') if path.is_file() and not path.name.endswith('.tmp')
             and (path.relative_to(ROOT).parts[0] in EXPORT_DIRS or path.relative_to(ROOT).as_posix() in allowed)]
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            bundle.writestr(MANIFEST, json.dumps({'plan': p}))
            for path in files:
                if path.is_symlink():
                    raise ValueError('Symlinks are not exported.')
                bundle.write(path, path.relative_to(ROOT).as_posix())
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def restore(archive):
    p = check_sources()
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())) or bundle.testzip() is not None:
            raise ValueError('Duplicate or damaged output archive.')
        saved_plan = read_zip_plan(bundle)
        if not same_experiment(saved_plan, p):
            raise ValueError('Different experiment: the historical 2048-token output cannot resume this run.')
        if sum(info.file_size for info in bundle.infolist()) > shutil.disk_usage(ROOT).free:
            raise OSError('Not enough space to restore the output archive.')
        for info in bundle.infolist():
            if info.filename == MANIFEST:
                continue
            if info.filename == 'plan.json':
                if json.loads(bundle.read(info)) != saved_plan:
                    raise ValueError('Archive plan differs from its manifest.')
                continue
            name = PurePosixPath(info.filename)
            if (info.orig_filename != info.filename or name.is_absolute() or '..' in name.parts
                or chr(92) in info.filename or ':' in info.filename or not name.parts
                or stat.S_ISLNK(info.external_attr >> 16)
                or not (name.parts[0] in EXPORT_DIRS or info.filename in
                        {'experiment_identity.json', 'plan.json', 'anchors.json', 'buckets_train_B440.json',
                         'buckets_train_B358.json', 'buckets_test_B440.json', 'comparison.json', 'REPORT.md'})):
                raise ValueError('Unsafe archive member.')
            target = ROOT.joinpath(*name.parts).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError('Unsafe archive destination.')
            if target.exists() and target.read_bytes() != bundle.read(info):
                raise ValueError('Existing progress differs; restore into a fresh runtime.')
        for info in bundle.infolist():
            if info.filename in (MANIFEST, 'plan.json') or info.is_dir():
                continue
            target = ROOT/info.filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.tmp')
            with bundle.open(info) as source, temporary.open('wb') as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(target)


def read_zip_plan(bundle):
    try:
        return json.loads(bundle.read(MANIFEST))['plan']
    except KeyError:
        raise ValueError('This archive is not from the Ngoc-matched Gemma notebook.') from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train', 'report', 'merge'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--merged')
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'train': train_one(args.arm, args.seed)
    elif args.command == 'report': report()
    elif args.command == 'merge': merge_adapter(args.checkpoint, args.merged)
