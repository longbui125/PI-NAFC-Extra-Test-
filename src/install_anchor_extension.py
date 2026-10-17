"""Install the verified anchor-pool recovery into an idle, existing Gemma Colab."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path('/content/gemma_ngoc')


def load_runner(path):
    spec = importlib.util.spec_from_file_location('gemma_anchor_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(revision):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Use the exact published recovery commit, not a moving branch.')
    required = ('plan.json', 'experiment_identity.json', 'anchors.json', 'buckets_train_B440.json',
                'buckets_train_B358.json', 'buckets_test_B440.json', 'results/base.json')
    if any(not (ROOT/name).exists() for name in required):
        raise ValueError('Restore the saved output first, or use the existing runtime containing your preparation.')
    processes = subprocess.check_output(['ps', '-eo', 'args='], text=True)
    if any(str(ROOT/'matched_run.py') in line.split() for line in processes.splitlines()):
        raise RuntimeError('A Gemma job is still active. Stop its cell before installing this recovery.')
    base = f'https://raw.githubusercontent.com/longbui125/PI-NAFC-Extra-Test-/{revision}/src/gemma_ngoc/'

    def fetch(name):
        with urllib.request.urlopen(base+name, timeout=60) as response:
            return response.read()

    plan_bytes = fetch('plan.json')
    new = json.loads(plan_bytes)
    payload = {}
    old = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    identity = json.loads((ROOT/'experiment_identity.json').read_text(encoding='utf-8'))
    for name, expected in new['source_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected source path in the published plan.')
        path = ROOT/name
        data = path.read_bytes() if path.exists() else b''
        if hashlib.sha256(data).hexdigest() != expected:
            if name in old['source_sha256'] and hashlib.sha256(data).hexdigest() != old['source_sha256'][name]:
                raise ValueError('Existing source was modified; refusing to overwrite it: '+name)
            data = fetch(name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Downloaded source hash mismatch: '+name)
        payload[name] = data
    with tempfile.TemporaryDirectory(prefix='nafc_recovery_', dir=ROOT) as directory:
        candidate = Path(directory)/'matched_run.py'
        candidate.write_bytes(payload['matched_run.py'])
        runner = load_runner(candidate)
        if not runner.same_experiment(old, new) or not runner.same_experiment(identity.get('plan'), new):
            raise ValueError('Existing output does not match the approved pre-training Olympic experiment.')
    if old['anchor_pool'] != new['anchor_pool'] and any((ROOT/'runs').rglob('*')):
        raise ValueError('Anchor-pool changes are not allowed after training has started.')
    protected = {'optimizer.pt', 'trainer_state.json', 'adapter_model.safetensors', 'pytorch_model.bin',
                 'model.safetensors', 'lambda_state.json', 'flip_state.json', 'run_info.json'}
    def saved_training_state(path):
        return path.name.startswith('checkpoint-') or path.name in protected or path.suffix in ('.safetensors', '.pt', '.pth', '.bin')
    if old['micro_batch'] != new['micro_batch'] and any(saved_training_state(p) for p in (ROOT/'runs').rglob('*')):
        raise ValueError('Saved training state exists; micro-batch migration may reuse preparation only. Nothing was overwritten.')
    previous = ROOT/'results/anchor_extension/previous_plan.json'
    if not previous.exists():
        runner.write(previous, old)
    for name, data in payload.items():
        path = ROOT/name
        temporary = path.with_suffix(path.suffix+'.upgrade.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
    runner.write(ROOT/'plan.json', new)
    runner.write(ROOT/'experiment_identity.json', {**identity, 'plan': new})
    runner = load_runner(ROOT/'matched_run.py')
    runner.check_sources()
    declared = {runner.run_name(arm, seed) for arm in new['arms'] for seed in new['seeds']}
    for folder in (ROOT/'runs').glob('*'):
        if folder.name not in declared or not folder.is_dir() or not any(folder.iterdir()):
            continue
        log = ROOT/'logs'/f'train_{folder.name}.log'
        if not log.exists():
            continue
        failed_log = log.read_text(encoding='utf-8', errors='replace')
        if not any(message in failed_log for message in ("Can't load image processor", 'torch.OutOfMemoryError')):
            continue
        if any(saved_training_state(p) for p in folder.rglob('*')):
            continue
        backup = ROOT/'results/failed_runs'/f'{folder.name}_{time.time_ns()}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not folder.resolve().is_relative_to((ROOT/'runs').resolve()):
            raise ValueError('Unexpected failed-initialization folder.')
        folder.rename(backup)
        print(f'Preserved failed-run metadata in {backup}. No saved optimizer checkpoint existed.', flush=True)
    print('Recovery installed: existing buckets, base evaluation and anchor results preserved.', flush=True)
    return runner
