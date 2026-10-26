"""Bounded subprocess supervision; never wait for pipe EOF from orphan workers."""
import codecs
import os
from pathlib import Path
import signal
import subprocess
import time
import re


def stop(process):
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run(command, log, *, timeout=3600, silence=900, heartbeat=30, cwd=None, total_steps=None):
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    started=last_output=last_notice=time.monotonic()
    decoder=codecs.getincrementaldecoder('utf8')(errors='replace')
    last_progress=started; highest_step=-1; tail=''
    # Regular files let the parent finish even when an orphan retains stdout.
    with log.open('ab',buffering=0) as writer, log.open('rb') as reader:
        reader.seek(0,os.SEEK_END)
        process=subprocess.Popen(command,cwd=cwd,stdout=writer,stderr=subprocess.STDOUT,
                                 start_new_session=os.name=='posix')
        print(f'[start] {log.name}; pid={process.pid}; timeout={timeout//60} min',flush=True)
        try:
            while True:
                chunk=reader.read(65536); now=time.monotonic()
                if chunk:
                    decoded=decoder.decode(chunk)
                    print(decoded,end='',flush=True); last_output=now
                    if total_steps is not None:
                        tail=(tail+decoded)[-8192:]
                        seen=[int(s) for s in re.findall(r'\b(\d+)/'+str(total_steps)+r'\b',tail)]
                        if seen and max(seen)>highest_step:
                            highest_step=max(seen); last_progress=now
                if process.poll() is not None:
                    break
                if now-started>timeout:
                    raise TimeoutError(f'Hard time limit reached; saved progress is preserved. Log: {log}')
                if now-last_output>silence:
                    raise TimeoutError(f'No child output for {silence:g}s; saved progress is preserved. Log: {log}')
                if total_steps is not None and now-last_progress>silence:
                    raise TimeoutError(f'No optimizer-step progress for {silence:g}s. Saved checkpoints are preserved: {log}')
                if now-last_notice>=heartbeat:
                    print(f'[running] {log.name}: {now-started:.0f}s elapsed, {now-last_output:.0f}s since child output',flush=True)
                    last_notice=now
                time.sleep(.15)
        finally:
            stop(process)
            remaining=os.fstat(reader.fileno()).st_size-reader.tell()
            while remaining>0:
                chunk=reader.read(min(remaining,65536))
                if not chunk: break
                print(decoder.decode(chunk),end='',flush=True); remaining-=len(chunk)
            print(decoder.decode(b'',final=True),end='',flush=True)
        if process.returncode:
            raise RuntimeError(f'Child failed ({process.returncode}). Original traceback is above; log: {log}')
    print(f'[done] {log.name}: {time.monotonic()-started:.1f}s',flush=True)


def selfcheck(folder):
    import sys
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    run([sys.executable,'-u','-c','print("supervisor ready",flush=True)'],folder/'guard_check.log',timeout=10)
    if os.name=='posix':
        code='import os,subprocess,sys; subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); print("orphan test",flush=True); os._exit(0)'
        started=time.monotonic()
        run([sys.executable,'-u','-c',code],folder/'orphan_check.log',timeout=10)
        if time.monotonic()-started>5:
            raise RuntimeError('Orphan cleanup self-check failed')
