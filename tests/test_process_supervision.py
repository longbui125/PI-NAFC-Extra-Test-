"""Real subprocess checks, without loading Torch or using GPU credits."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from test_gemma_ngoc import load_runner, runner


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    module = load_runner()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'POLL_INTERVAL', .02)
    parents = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        parents.append(process)
        return process

    monkeypatch.setattr(subprocess, 'Popen', spawn)
    if os.name != 'posix':
        monkeypatch.setattr(module.signal, 'SIGKILL', 9, raising=False)
        # Colab uses killpg; terminate only this test's known processes on Windows.
        def kill_group(pid, sig):
            assert pid in {p.pid for p in parents}
            for path in tmp_path.glob('worker*.pid'):
                try:
                    os.kill(int(path.read_text()), signal.SIGTERM)
                except OSError:
                    pass
            for process in parents:
                if process.pid == pid and process.poll() is None:
                    process.kill()
        monkeypatch.setattr(module.os, 'killpg', kill_group, raising=False)
    try:
        yield module
    finally:
        for path in tmp_path.glob('worker*.pid'):
            try:
                os.kill(int(path.read_text()), signal.SIGTERM)
            except OSError:
                pass
        for process in parents:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_parent_exit_does_not_wait_for_orphan_holding_stdout(supervisor, tmp_path, capsys):
    m = supervisor
    log = tmp_path/'orphan.log'
    log.write_bytes(b'previous completed output\n')
    worker_pid = tmp_path/'worker.pid'
    code = ('import os,subprocess,sys; from pathlib import Path; '
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
            f'Path({str(worker_pid)!r}).write_text(str(p.pid)); '
            'print("result saved",flush=True);os._exit(0)')
    started = time.monotonic()
    m.run_logged([sys.executable, '-u', '-c', code], log)
    assert time.monotonic() - started < 5
    assert log.read_text().startswith('previous completed output\nresult saved')
    output = capsys.readouterr().out
    assert 'result saved' in output and '[done]' in output
    assert 'previous completed output' not in output


def test_silent_live_process_times_out_preserving_saved_files(supervisor, tmp_path, monkeypatch, capsys):
    m = supervisor
    monkeypatch.setattr(m, 'SILENCE_LIMIT', .3)
    monkeypatch.setattr(m, 'HEARTBEAT_INTERVAL', .05)
    saved = tmp_path/'completed.json'; saved.write_text('{"saved":true}')
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='Completed output/checkpoints are preserved'):
        m.run_logged([sys.executable, '-u', '-c', 'import time;time.sleep(30)'], tmp_path/'silent.log')
    assert time.monotonic() - started < 5
    assert saved.read_text() == '{"saved":true}'
    assert '[running]' in capsys.readouterr().out


def test_failed_child_reports_original_stderr_and_nonzero_status(supervisor, tmp_path, capsys):
    m = supervisor
    with pytest.raises(RuntimeError, match=r'Command failed \(7\)'):
        m.run_logged([sys.executable, '-u', '-c',
                      'import sys;print("original failure",file=sys.stderr);sys.exit(7)'], tmp_path/'error.log')
    assert 'original failure' in capsys.readouterr().out
    assert 'original failure' in (tmp_path/'error.log').read_text()


def test_live_log_handles_split_utf8_without_newline(supervisor, tmp_path, capsys):
    m = supervisor
    m.run_logged([sys.executable, '-u', '-c',
                  'import os,time;os.write(1,b"\\xe2");time.sleep(.15);os.write(1,b"\\x82\\xac")'],
                 tmp_path/'utf8.log')
    assert '\u20ac' in capsys.readouterr().out
    assert (tmp_path/'utf8.log').read_bytes() == '\u20ac'.encode()


def test_keyboard_interrupt_cleans_up_child(supervisor, tmp_path, monkeypatch):
    original_sleep = time.sleep
    interrupted = False
    def interrupt(_):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_sleep(_)
    monkeypatch.setattr(supervisor.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        supervisor.run_logged([sys.executable, '-u', '-c', 'import time;time.sleep(30)'], tmp_path/'interrupt.log')


@pytest.mark.skipif(os.name != 'posix', reason='Linux process groups; also runs in the Colab setup cell')
def test_real_linux_orphan_cleanup_before_gpu_work():
    load_runner().check_process_cleanup()


def test_restore_previous_olympic_zip_preserves_data_and_keeps_new_runner_plan(runner, tmp_path):
    m = runner
    current = m.check_sources()
    legacy = json.loads((Path(__file__).parent/'fixtures/gemma_pool800_plan.json').read_text(encoding='utf-8'))
    import zipfile
    archive = tmp_path.parent/'before_process_fix.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr(m.MANIFEST, json.dumps({'plan': legacy}))
        bundle.writestr('plan.json', json.dumps(legacy))
        bundle.writestr('buckets_test_B440.json', '{"saved":true}')
        bundle.writestr('experiment_identity.json', json.dumps({'plan': legacy, 'software': {'v': '1'}}))
    m.restore(archive)
    assert m.read(m.ROOT/'plan.json') == current
    assert m.read(m.ROOT/'buckets_test_B440.json') == {'saved': True}
    assert m.same_experiment(m.read(m.ROOT/'experiment_identity.json')['plan'], current)
    legacy['max_tokens'] = 2048
    assert not m.same_experiment(legacy, current)
    legacy['max_tokens'] = 1536
    legacy['source_sha256']['train.py'] = 'changed'
    assert not m.same_experiment(legacy, current)


def test_live_guard_cleans_only_completed_groups_owned_by_gemma(monkeypatch):
    source = Path(__file__).resolve().parents[1]/'src/colab_process_guard.py'
    spec = importlib.util.spec_from_file_location('live_gemma_guard', source)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    killed = []
    monkeypatch.setattr(guard.signal, 'SIGKILL', 9, raising=False)
    monkeypatch.setattr(guard.os, 'killpg', lambda p, s: killed.append(p), raising=False)
    root = (9, 1, 1, 'Ssl', f'python -u {guard.RUNNER} prepare')
    rows = {1: root, 2: (1, 2, 2, 'R', 'python evaluate.py'),
            3: (1, 3, 3, 'Z', '[python] <defunct>'),
            4: (99, 4, 4, 'Z', 'unrelated'), 5: (1, 1, 1, 'Z', 'not a new session')}
    tracked = set()
    assert guard.reap_finished(rows, tracked) == {1}
    assert killed == [3] and tracked == {(1, 2)}
    guard.reap_finished({1: root, 7: (1, 1, 1, 'S', 'argument with unmatched "')}, tracked)
    assert killed == [3, 2] and not tracked


def test_nested_cancellation_collects_only_descendant_session_leaders(monkeypatch):
    m = load_runner()
    rows = '10 1 10 10\n20 10 20 20\n21 20 20 20\n30 21 30 30\n99 1 99 99\n'
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: rows)
    assert m.descendant_sessions(10) == {10, 20, 30}


def test_live_guard_times_out_without_deleting_saved_output(tmp_path, monkeypatch):
    import types
    source = Path(__file__).resolve().parents[1]/'src/colab_process_guard.py'
    spec = importlib.util.spec_from_file_location('timed_gemma_guard', source)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    path = tmp_path/'matched_run.py'; path.write_text('saved runner')
    saved = tmp_path/'buckets_test_B440.json'; saved.write_text('saved output')
    monkeypatch.setattr(guard, 'RUNNER', str(path))
    rows = {1: (9, 1, 1, 'Ssl', f'python -u {path} prepare'),
            2: (1, 2, 2, 'R', 'python evaluate.py'), 99: (9, 99, 99, 'R', 'unrelated')}
    killed = []
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(guard, 'snapshot', lambda: rows)
    monkeypatch.setattr(guard, 'time', types.SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None))
    monkeypatch.setattr(guard, 'os', types.SimpleNamespace(name='posix', killpg=lambda p, s: killed.append(p)))
    monkeypatch.setattr(guard.signal, 'SIGKILL', 9, raising=False)
    monkeypatch.setattr(guard, 'SILENCE_LIMIT', 1)
    with pytest.raises(TimeoutError, match='saved files are untouched'):
        guard.main()
    assert set(killed) == {1, 2} and saved.read_text() == 'saved output'


def test_old_identity_migrates_runner_only_and_rejects_environment_changes(runner, monkeypatch):
    import types
    m = runner; p = m.check_sources()
    gpu = 'NVIDIA A100-SXM4-40GB'
    software = {**p['versions'], 'math-verify': 'test', 'datasets': 'test'}
    legacy = json.loads((Path(__file__).parent/'fixtures/gemma_pool800_plan.json').read_text(encoding='utf-8'))
    identity = m.ROOT/'experiment_identity.json'
    m.write(identity, {'plan': legacy, 'software': software, 'gpu': gpu})
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: gpu)))
    monkeypatch.setattr(m.importlib.metadata, 'version', lambda name: software[name])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=lambda *a, **kw: None))
    assert m.initialize() == p and m.read(identity)['plan'] == p
    wrong = m.read(identity); wrong['gpu'] = 'different GPU'; m.write(identity, wrong)
    with pytest.raises(ValueError, match='different model, configuration or environment'):
        m.initialize()
