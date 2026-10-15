"""Numeric-answer Math-Verify scoring, with bounded Windows worker calls."""
import atexit
import json
import os
import queue
import re
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from math_verify import parse, verify
from core import last_boxed, _normalize

_worker = None


def _parse_numeric(answer, timeout):
    if not answer or len(answer) > 256:
        return None
    # Reject prose, symbolic variables and multiple answers before any extraction.
    plain = re.sub(r"\\(?:dfrac|tfrac|frac|sqrt|pi|cdot|times|left|right|boxed)\b", "", answer)
    if re.search(r"[A-Za-z,;=<>]", plain) or re.search(r"[^0-9\s.${}()\[\]+*/^!\-\\]", plain):
        return None
    parsed = parse(f"${answer.strip('$')}$", fallback_mode="no_fallback", parsing_timeout=timeout)
    value = next((x for x in parsed if not isinstance(x, str)), None)
    return value if getattr(value, "is_number", None) is True and getattr(value, "is_finite", None) is True else None


def _compute(request, timeout):
    if request['op'] == 'verify_math':
        if _normalize(request['gold']) == _normalize(request['pred']):
            return True
        gold = parse(f"${request['gold']}$", fallback_mode='no_fallback', parsing_timeout=timeout)
        pred = parse(f"${request['pred']}$", fallback_mode='no_fallback', parsing_timeout=timeout)
        return bool(gold and pred and verify(gold, pred, timeout_seconds=timeout))
    gold = _parse_numeric(request["gold"], timeout)
    if request["op"] == "valid":
        return gold is not None
    pred = _parse_numeric(request["pred"], timeout)
    return gold is not None and pred is not None and bool(verify(gold, pred, timeout_seconds=timeout))


def _stop_worker():
    global _worker
    if _worker is not None:
        if _worker.poll() is None:
            _worker.kill()
        _worker.wait()
        _worker.stdin.close(); _worker.stdout.close()
        _worker = None


def _bounded(request):
    global _worker
    if os.name != "nt":
        return _compute(request, 5)
    # Math-Verify's Windows timeout spawns a non-picklable nested function.
    # A persistent plain subprocess avoids that bug; the parent enforces the deadline.
    if _worker is None:
        _worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8")
    inbox = queue.Queue()
    try:
        _worker.stdin.write(json.dumps(request)+"\n"); _worker.stdin.flush()
        threading.Thread(target=lambda: inbox.put(_worker.stdout.readline()), daemon=True).start()
        response = inbox.get(timeout=10)
        if not response:
            raise RuntimeError("Scoring worker exited")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]
    except Exception as exc:
        _stop_worker()
        raise RuntimeError("Answer verification failed or timed out; no false reward was assigned.") from exc


@lru_cache(maxsize=32768)
def numeric_gold(answer):
    return answer if _bounded({"op": "valid", "gold": answer}) else None


def is_correct(completion, gold):
    pred = last_boxed(completion)
    if pred is None or numeric_gold(gold) is None or numeric_gold(pred) is None:
        return False
    return _bounded({"op": "verify", "gold": gold, "pred": pred})


def math_is_correct(completion, gold):
    pred = last_boxed(completion)
    return pred is not None and bool(_bounded({'op': 'verify_math', 'gold': gold, 'pred': pred}))


def math_gold(solution):
    answer = last_boxed(solution)
    if answer is None:
        unbraced = re.findall(r'\\boxed\s+([+-]?\d+(?:\.\d+)?)(?=\s|\$|[.,]|$)', solution)
        answer = unbraced[-1] if unbraced else None
    return answer


atexit.register(_stop_worker)


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    for line in sys.stdin:
        try:
            result = {"result": _compute(json.loads(line), None)}
        except Exception as exc:
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)
