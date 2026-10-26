import os
import sys
import time

import pytest

import guard


def test_supervisor_reads_logs_and_reports_child_errors(tmp_path,capsys):
    log=tmp_path/'job.log'
    guard.run([sys.executable,'-u','-c','print("completed")'],log,timeout=10,heartbeat=.1)
    assert 'completed' in log.read_text() and '[done]' in capsys.readouterr().out
    with pytest.raises(RuntimeError,match='Child failed'):
        guard.run([sys.executable,'-c','raise ValueError("test_failure")'],log,timeout=10)
    assert 'test_failure' in log.read_text()


def test_silent_process_times_out_without_touching_saved_checkpoint(tmp_path):
    saved=tmp_path/'checkpoint';saved.write_text('saved')
    with pytest.raises(TimeoutError,match='No child output'):
        guard.run([sys.executable,'-c','import time;time.sleep(30)'],
                  tmp_path/'silent.log',timeout=10,silence=.3,heartbeat=.1)
    assert saved.read_text()=='saved'


@pytest.mark.skipif(os.name!='posix',reason='Colab process groups require Linux')
def test_parent_exit_does_not_wait_for_orphan_stdout(tmp_path):
    started=time.monotonic()
    script='import os,subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);print("saved",flush=True);os._exit(0)'
    guard.run([sys.executable,'-u','-c',script],tmp_path/'orphan.log',timeout=5)
    assert time.monotonic()-started<5
