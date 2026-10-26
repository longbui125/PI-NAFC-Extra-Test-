"""Frozen preparation helpers retained from the upstream experiment."""
import random


def bucket_of(p: float, thresholds: list[float]) -> int:
    """thresholds giảm dần, ví dụ [0.75, 0.25]: p>=0.75 -> 0 (dễ), p>=0.25 -> 1 (vừa), còn lại -> 2 (khó)."""
    for k, t in enumerate(thresholds):
        if p >= t:
            return k
    return len(thresholds)


def pick_anchors(scored: list[dict], n_anchor: int, hi: float, seed: int) -> list[dict]:
    cand = [x for x in scored if x["p_select"] >= hi and x["base_response"] is not None]
    rng = random.Random(seed)
    rng.shuffle(cand)
    return cand[:n_anchor]


def noise_floors(anchors: list[dict], lo: float) -> dict:
    n = max(len(anchors), 1)
    return {"noise_robust": sum(x["p_check"] <= lo for x in anchors) / n,
            "noise_raw": sum(x["base_first_correct"] and not x["check_first_correct"] for x in anchors) / n}
