"""Release finished Gemma child sessions in an already-running, unpatched Colab."""
import os
from pathlib import Path
import signal
import subprocess
import time

RUNNER = '/content/gemma_ngoc/matched_run.py'
SILENCE_LIMIT = 15 * 60


def snapshot():
    output = subprocess.check_output(
        ['ps', '-eo', 'pid=,ppid=,pgid=,sid=,stat=,args='], text=True)
    return {int(p): (int(parent), int(group), int(session), state, args)
            for line in output.splitlines() if line.strip()
            for p, parent, group, session, state, args in [line.split(None, 5)]}


def reap_finished(rows, tracked):
    owners = {pid for pid, row in rows.items() if RUNNER in row[4].split()}
    for pid, (parent, group, session, state, args) in rows.items():
        if parent in owners and pid == group == session:
            tracked.add((parent, group))
    for owner, group in list(tracked):
        leader = rows.get(group)
        if owner not in owners or leader is None or leader[3].startswith(('Z', 'X')):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            tracked.remove((owner, group))
            print(f'[cleanup] Finished child session {group}; completed files are untouched.', flush=True)
    return owners


def main():
    if os.name != 'posix' or not Path(RUNNER).exists():
        raise RuntimeError('Run this helper in the existing Gemma Colab Terminal.')
    tracked = set()
    idle = heartbeat = last_output = time.monotonic()
    previous = None
    print('Watching existing Gemma jobs. Only their own child sessions are eligible for cleanup. Ctrl+C exits.',
          flush=True)
    while True:
        rows = snapshot()
        owners = reap_finished(rows, tracked)
        now = time.monotonic()
        logs = Path(RUNNER).parent/'logs'
        progress = (sorted(owners), sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                                          for p in logs.glob('*.log')))
        if progress != previous:
            last_output, previous = now, progress
        if owners and now - last_output >= SILENCE_LIMIT:
            groups = {group for owner, group in tracked}
            groups.update(pid for pid in owners if pid == rows[pid][1] == rows[pid][2])
            for group in sorted(groups, reverse=True):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise TimeoutError('No log progress for 15 minutes. Gemma jobs stopped; saved files are untouched.')
        if owners:
            idle = now
        elif now - idle >= 60:
            print('No Gemma job for 60 seconds; helper finished.', flush=True)
            return
        if now - heartbeat >= 60:
            print(f'[watching] jobs={sorted(owners)}; {now-last_output:.0f}s since log progress; '
                  f'child sessions={sorted(tracked)}', flush=True)
            heartbeat = now
        time.sleep(1)


if __name__ == '__main__':
    main()
