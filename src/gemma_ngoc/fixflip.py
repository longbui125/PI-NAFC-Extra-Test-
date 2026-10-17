"""
fixflip.py — Module đo chung của đồ án: học thêm (sai → đúng) và làm hỏng (đúng → sai), có mức nhiễu.

Chỉ cần numpy. Dùng chung cho cả ba phần:
  - A, nhánh RL (Qwen, MATH, lấy mẫu T = 1, k mẫu mỗi câu),
  - C, nhánh SFT (OLMoE, HotpotQA; giải mã greedy thì k = 1),
  - bất kỳ cặp "model gốc / model sau train" nào chấm trên CÙNG một danh sách câu, CÙNG thứ tự.

Đầu vào chuẩn là `Scores`: với mỗi câu, tỷ lệ đúng p̂ trên k mẫu (cùng k cho mọi câu), tuỳ chọn độ dài
trung bình và nhãn nhóm (ví dụ nhóm độ khó). Đọc được file của evaluate.py, prepare_buckets.py, hoặc JSONL chung.

Các đại lượng (θ₀ = model gốc, θ = model mới, p₀/p₁ = tỷ lệ đúng từng câu):
  - Kỳ vọng 1 mẫu:   fix  = E[(1 − p₀) p₁] / E[1 − p₀]
                      flip = E[p₀ (1 − p₁)] / E[p₀]
    tức xác suất một mẫu của θ sai trên câu mà một mẫu của θ₀ đúng (flip), và ngược lại (fix).
    Luôn có  ΔAcc = E[p₁] − E[p₀] = E[(1 − p₀) p₁] − E[p₀ (1 − p₁)].
  - Chắc chắn:       câu hỏng thật   nếu p₀ ≥ hi và p₁ ≤ lo (mặc định 0,75 / 0,25), chia cho số câu có p₀ ≥ hi;
                      câu học thêm thật nếu p₀ ≤ lo và p₁ ≥ hi, chia cho số câu có p₀ ≤ lo.
  - Mức nhiễu:       cùng phép đo giữa hai lần chấm độc lập của CÙNG một model (model gốc chấm lại khi lấy mẫu,
                      hoặc hai seed cùng cấu hình khi greedy). Câu hỏng "vượt nhiễu" = flip − flip của mức nhiễu.

Ví dụ (dữ liệu phần A):
    base  = Scores.from_buckets_json("buckets_test_B440.json")     # model gốc, 8 mẫu, seed 0
    base2 = Scores.from_eval_json("results/base.json")             # model gốc chấm lại, 4 mẫu, seed 1
    noise = fixflip(base, base2)          # flip ≈ 0,391 (kỳ vọng 1 mẫu), robust_flip ≈ 0,039
    run   = Scores.from_eval_json("results/nafc_s0.json")
    print(fixflip(base, run))
    print(compare(base, run, Scores.from_eval_json("results/grpo_s0.json")))   # chênh lệch + KTC 95%

Dòng lệnh:
    python fixflip.py --base buckets_test_B440.json --base2 results/base.json \
        --runs results/grpo_s0.json results/nafc_s0.json --ref results/grpo_s0.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

HI, LO = 0.75, 0.25


# --------------------------------------------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------------------------------------------- #
@dataclass
class Scores:
    """Kết quả chấm của MỘT model trên một danh sách câu cố định.

    p:      tỷ lệ đúng từng câu, trong [0, 1] (trung bình trên k mẫu)
    k:      số mẫu mỗi câu (1 nếu greedy)
    length: độ dài trung bình từng câu (token), tuỳ chọn
    group:  nhãn nhóm từng câu (int), tuỳ chọn
    ids:    mã câu, tuỳ chọn; nếu cả hai Scores đều có ids thì phép so kiểm tra khớp thứ tự
    """

    p: np.ndarray
    k: int = 1
    length: np.ndarray | None = None
    group: np.ndarray | None = None
    ids: list | None = None
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=float)
        if self.p.ndim != 1:
            raise ValueError("p phải là mảng 1 chiều (một giá trị mỗi câu)")
        if np.any((self.p < 0) | (self.p > 1)):
            raise ValueError("p phải nằm trong [0, 1]")
        if self.length is not None:
            self.length = np.asarray(self.length, dtype=float)
            assert self.length.shape == self.p.shape, "length phải cùng số câu với p"
        if self.group is not None:
            self.group = np.asarray(self.group)
            assert self.group.shape == self.p.shape, "group phải cùng số câu với p"
        if self.ids is not None:
            assert len(self.ids) == len(self.p), "ids phải cùng số câu với p"

    # ---- thống kê đơn ----
    def __len__(self):
        return len(self.p)

    @property
    def acc(self) -> float:
        return float(self.p.mean())

    @property
    def mean_len(self) -> float:
        return float(self.length.mean()) if self.length is not None else float("nan")

    def take(self, idx) -> "Scores":
        """Chọn một tập con câu (mảng bool hoặc chỉ số); dùng cho bootstrap và chia nhóm."""
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return Scores(p=self.p[idx], k=self.k,
                      length=None if self.length is None else self.length[idx],
                      group=None if self.group is None else self.group[idx],
                      ids=None if self.ids is None else [self.ids[i] for i in idx],
                      name=self.name, extra=self.extra)

    # ---- đọc file ----
    @classmethod
    def from_correct(cls, correct, length=None, group=None, ids=None, name="") -> "Scores":
        """Từ ma trận đúng/sai (số câu × k mẫu)."""
        c = np.asarray(correct, dtype=float)
        if c.ndim == 1:
            c = c[:, None]
        L = None if length is None else np.asarray(length, dtype=float)
        if L is not None and L.ndim == 2:
            L = L.mean(1)
        return cls(p=c.mean(1), k=c.shape[1], length=L, group=group, ids=ids, name=name)

    @classmethod
    def from_eval_json(cls, path: str, name: str | None = None) -> "Scores":
        """File của evaluate.py: per_question[i] = {acc, len, bucket, ...}; decoding.n = k."""
        d = json.load(open(path))
        q = d["per_question"]
        return cls(p=[x["acc"] for x in q], k=int(d.get("decoding", {}).get("n", 1)),
                   length=[x["len"] for x in q] if "len" in q[0] else None,
                   group=[x["bucket"] for x in q] if "bucket" in q[0] else None,
                   ids=[x["id"] for x in q] if "id" in q[0] else None,
                   name=name or path)

    @classmethod
    def from_buckets_json(cls, path: str, name: str | None = None) -> "Scores":
        """File của prepare_buckets.py: items[i] = {p, bucket, samples: [[đúng, độ dài], ...]}."""
        d = json.load(open(path))
        it = d["items"]
        return cls(p=[x["p"] for x in it], k=int(d.get("n", d.get("meta", {}).get("n", 1))),
                   length=[float(np.mean([s[1] for s in x["samples"]])) for x in it] if "samples" in it[0] else None,
                   group=[x["bucket"] for x in it] if "bucket" in it[0] else None,
                   name=name or path)

    @classmethod
    def from_jsonl(cls, path: str, name: str | None = None) -> "Scores":
        """JSONL chung, mỗi dòng một câu: {"id": ..., "correct": [0/1, ...], "length": [...]?, "group": ...?}.
        Định dạng khuyên dùng cho nhánh SFT (ghi ra từ script đánh giá của bất kỳ framework nào)."""
        rows = [json.loads(line) for line in open(path) if line.strip()]
        ks = {len(r["correct"]) for r in rows}
        if len(ks) != 1:
            raise ValueError(f"mọi câu phải có cùng số mẫu, thấy {sorted(ks)}")
        return cls.from_correct([r["correct"] for r in rows],
                                length=[np.mean(r["length"]) for r in rows] if "length" in rows[0] else None,
                                group=[r["group"] for r in rows] if "group" in rows[0] else None,
                                ids=[r["id"] for r in rows] if "id" in rows[0] else None,
                                name=name or path)

    @classmethod
    def load(cls, path: str, name: str | None = None) -> "Scores":
        if path.endswith(".jsonl"):
            return cls.from_jsonl(path, name)
        d = json.load(open(path))
        if "per_question" in d:
            return cls.from_eval_json(path, name)
        if "items" in d:
            return cls.from_buckets_json(path, name)
        raise ValueError(f"không nhận ra định dạng file {path}")


def _check_pair(a: Scores, b: Scores):
    if len(a) != len(b):
        raise ValueError(f"hai model chấm trên số câu khác nhau: {len(a)} và {len(b)} ({a.name} / {b.name})")
    if a.ids is not None and b.ids is not None and list(a.ids) != list(b.ids):
        raise ValueError(f"thứ tự câu không khớp giữa {a.name} và {b.name}")


# --------------------------------------------------------------------------------------------------------------- #
# Đo học thêm / làm hỏng
# --------------------------------------------------------------------------------------------------------------- #
def per_question(base: Scores, new: Scores, hi: float = HI, lo: float = LO) -> dict[str, np.ndarray]:
    """Các đại lượng TỪNG CÂU; mọi tỷ lệ tổng hợp = tổng tử số / tổng mẫu số. Dùng lại cho bootstrap."""
    _check_pair(base, new)
    p0, p1 = base.p, new.p
    return {
        "acc0": p0, "acc1": p1,
        "fix_num": (1 - p0) * p1, "fix_den": 1 - p0,                       # kỳ vọng 1 mẫu
        "flip_num": p0 * (1 - p1), "flip_den": p0,
        "rfix_num": ((p0 <= lo) & (p1 >= hi)).astype(float), "rfix_den": (p0 <= lo).astype(float),   # chắc chắn
        "rflip_num": ((p0 >= hi) & (p1 <= lo)).astype(float), "rflip_den": (p0 >= hi).astype(float),
    }


def _ratio(num, den):
    s = float(np.sum(den))
    return float(np.sum(num)) / s if s > 0 else float("nan")


def summarize_pq(q: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "acc_base": float(np.mean(q["acc0"])), "acc_new": float(np.mean(q["acc1"])),
        "delta": float(np.mean(q["acc1"]) - np.mean(q["acc0"])),
        "fix": _ratio(q["fix_num"], q["fix_den"]), "flip": _ratio(q["flip_num"], q["flip_den"]),
        "robust_fix": _ratio(q["rfix_num"], q["rfix_den"]), "robust_flip": _ratio(q["rflip_num"], q["rflip_den"]),
        "n_robust_fix": int(np.sum(q["rfix_num"])), "n_base_wrong": int(np.sum(q["rfix_den"])),
        "n_robust_flip": int(np.sum(q["rflip_num"])), "n_base_right": int(np.sum(q["rflip_den"])),
    }


def fixflip(base: Scores, new: Scores, hi: float = HI, lo: float = LO) -> dict[str, float]:
    """Học thêm / làm hỏng của `new` so với `base`, cả bản kỳ vọng 1 mẫu và bản chắc chắn."""
    out = summarize_pq(per_question(base, new, hi, lo))
    out["mean_len"] = new.mean_len
    return out


def noise_floor(scoring_a: Scores, scoring_b: Scores, hi: float = HI, lo: float = LO) -> dict[str, float]:
    """Mức nhiễu = fixflip giữa hai lần chấm độc lập của CÙNG một model."""
    return fixflip(scoring_a, scoring_b, hi, lo)


def by_group(base: Scores, new: Scores, hi: float = HI, lo: float = LO, groups=None) -> dict:
    """fixflip trong từng nhóm (mặc định nhóm của base, ví dụ nhóm độ khó dễ/vừa/khó)."""
    g = np.asarray(groups if groups is not None else base.group)
    if g is None or g.ndim == 0:
        raise ValueError("cần nhãn nhóm (base.group hoặc tham số groups)")
    return {k.item() if hasattr(k, "item") else k: fixflip(base.take(g == k), new.take(g == k), hi, lo)
            for k in np.unique(g)}


# --------------------------------------------------------------------------------------------------------------- #
# Bootstrap ghép cặp theo câu
# --------------------------------------------------------------------------------------------------------------- #
METRICS = ("acc_new", "delta", "fix", "flip", "robust_fix", "robust_flip")
_NUM_DEN = {"fix": ("fix_num", "fix_den"), "flip": ("flip_num", "flip_den"),
            "robust_fix": ("rfix_num", "rfix_den"), "robust_flip": ("rflip_num", "rflip_den")}


def _boot_idx(n: int, B: int, seed: int, chunk: int = 500):
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        yield rng.integers(0, n, size=(m, n))
        done += m


def _metric_from_pq(q, metric, idx=None):
    def S(key):
        v = q[key]
        return v.sum() if idx is None else v[idx].sum(axis=-1)
    if metric == "acc_new":
        n = len(q["acc1"]) if idx is None else idx.shape[-1]
        return S("acc1") / n
    if metric == "delta":
        n = len(q["acc1"]) if idx is None else idx.shape[-1]
        return (S("acc1") - S("acc0")) / n
    num, den = _NUM_DEN[metric]
    d = S(den)
    return np.where(d > 0, S(num) / np.where(d > 0, d, 1), np.nan)


def bootstrap_ci(base: Scores, new: Scores, metric: str = "robust_flip", B: int = 4000, seed: int = 0,
                 hi: float = HI, lo: float = LO, alpha: float = 0.05) -> tuple[float, float, float]:
    """(ước lượng, cận dưới, cận trên) của một đại lượng, KTC bootstrap theo câu."""
    q = per_question(base, new, hi, lo)
    est = float(_metric_from_pq(q, metric))
    vals = np.concatenate([_metric_from_pq(q, metric, idx) for idx in _boot_idx(len(base), B, seed)])
    lo_, hi_ = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo_), float(hi_)


def compare(base: Scores, x: Scores, y: Scores, metrics: Sequence[str] = METRICS, B: int = 4000, seed: int = 0,
            hi: float = HI, lo: float = LO, alpha: float = 0.05) -> dict[str, tuple[float, float, float]]:
    """Chênh lệch x − y (cùng base, cùng câu) cho từng đại lượng, với KTC bootstrap GHÉP CẶP:
    mỗi lần lấy lại mẫu dùng CÙNG tập câu cho cả x và y."""
    qx, qy = per_question(base, x, hi, lo), per_question(base, y, hi, lo)
    out = {}
    for m in metrics:
        est = float(_metric_from_pq(qx, m) - _metric_from_pq(qy, m))
        vals = np.concatenate([_metric_from_pq(qx, m, idx) - _metric_from_pq(qy, m, idx)
                               for idx in _boot_idx(len(base), B, seed)])
        lo_, hi_ = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = (est, float(lo_), float(hi_))
    return out


# --------------------------------------------------------------------------------------------------------------- #
# So ở cùng chi phí: nội suy từng câu giữa hai điểm kẹp hai bên
# --------------------------------------------------------------------------------------------------------------- #
def iso_interp(frontier: Sequence[Scores], target: float,
               cost: Callable[[Scores], float] = lambda s: s.mean_len) -> tuple[Scores, dict]:
    """Model "ảo" trên đường nối các điểm `frontier`, tại chi phí `target`.

    Nội suy tuyến tính TỪNG CÂU giữa hai điểm kẹp hai bên (như phần A so ở cùng độ dài). Không ngoại suy:
    ngoài khoảng thì lấy điểm đầu mút và báo `clamped`. Độ chính xác và học thêm / làm hỏng KỲ VỌNG 1 mẫu
    tuyến tính theo p₁ nên nội suy được chính xác; bản chắc chắn thì nên nội suy chính các tỷ lệ (interp_scalar).

    cost: hàm chi phí của một điểm, mặc định độ dài trung bình; dùng `lambda s: s.acc` để so ở cùng độ chính xác.
    """
    pts = sorted(frontier, key=cost)
    cs = [cost(s) for s in pts]
    for a, b in zip(pts, pts[1:]):
        _check_pair(a, b)
    if target <= cs[0] or target >= cs[-1]:
        j = 0 if target <= cs[0] else len(pts) - 1
        s = pts[j]
        return Scores(p=s.p.copy(), k=s.k, length=None if s.length is None else s.length.copy(), group=s.group,
                      ids=s.ids, name=f"{s.name}@{target:.3g}"), {"lo": j, "hi": j, "w": 0.0, "clamped": True}
    j = int(np.searchsorted(cs, target))
    w = (target - cs[j - 1]) / (cs[j] - cs[j - 1])
    a, b = pts[j - 1], pts[j]
    length = None if a.length is None or b.length is None else (1 - w) * a.length + w * b.length
    s = Scores(p=(1 - w) * a.p + w * b.p, k=min(a.k, b.k), length=length, group=a.group, ids=a.ids,
               name=f"interp({a.name},{b.name})@{target:.3g}")
    return s, {"lo": j - 1, "hi": j, "w": float(w), "clamped": False}


def interp_scalar(costs: Sequence[float], values: Sequence[float], target: float) -> float:
    """Nội suy tuyến tính một đại lượng vô hướng theo chi phí (không ngoại suy)."""
    order = np.argsort(costs)
    c, v = np.asarray(costs, float)[order], np.asarray(values, float)[order]
    return float(np.interp(target, c, v))


# --------------------------------------------------------------------------------------------------------------- #
# In bảng
# --------------------------------------------------------------------------------------------------------------- #
def pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def table(base: Scores, runs: Sequence[Scores], noise: dict | None = None, hi: float = HI, lo: float = LO) -> str:
    """Bảng gọn: accuracy, độ dài, học thêm / làm hỏng (kỳ vọng và chắc chắn), làm hỏng vượt nhiễu."""
    hdr = (f"{'run':28s} {'acc':>6s} {'len':>6s} {'Δacc':>6s} {'fix':>6s} {'flip':>6s} "
           f"{'rfix':>6s} {'rflip':>6s} {'vượt nhiễu':>10s}")
    lines = [hdr, "-" * len(hdr)]
    for r in runs:
        m = fixflip(base, r, hi, lo)
        ex = m["robust_flip"] - noise["robust_flip"] if noise else float("nan")
        lines.append(f"{(r.name or '?')[-28:]:28s} {pct(m['acc_new']):>6s} {r.mean_len:6.0f} {pct(m['delta']):>6s} "
                     f"{pct(m['fix']):>6s} {pct(m['flip']):>6s} {pct(m['robust_fix']):>6s} {pct(m['robust_flip']):>6s} "
                     f"{pct(ex):>10s}")
    if noise:
        lines.append(f"{'mức nhiễu (gốc chấm lại)':28s} {'':6s} {'':6s} {'':6s} {pct(noise['fix']):>6s} "
                     f"{pct(noise['flip']):>6s} {pct(noise['robust_fix']):>6s} {pct(noise['robust_flip']):>6s}")
    lines.append(f"(%; fix/flip = kỳ vọng 1 mẫu; rfix/rflip = chắc chắn, ngưỡng {hi}/{lo})")
    return "\n".join(lines)


def fmt_compare(res: dict[str, tuple[float, float, float]], label: str = "") -> str:
    parts = [f"{m} {100 * e:+.2f} [{100 * l:+.2f}; {100 * h:+.2f}]" for m, (e, l, h) in res.items()]
    return (label + ": " if label else "") + " | ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Bảng học thêm / làm hỏng có mức nhiễu")
    ap.add_argument("--base", required=True, help="model gốc (vd buckets_test_B440.json, 8 mẫu)")
    ap.add_argument("--base2", default=None, help="model gốc chấm lại độc lập (vd results/base.json) để đo mức nhiễu")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="so từng run với run này (vd GRPO cùng seed), KTC bootstrap ghép cặp")
    ap.add_argument("--hi", type=float, default=HI)
    ap.add_argument("--lo", type=float, default=LO)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    base = Scores.load(args.base, "base")
    noise = noise_floor(base, Scores.load(args.base2, "base2"), args.hi, args.lo) if args.base2 else None
    runs = [Scores.load(p) for p in args.runs]
    print(table(base, runs, noise, args.hi, args.lo))
    if args.ref:
        ref = Scores.load(args.ref)
        for r in runs:
            if r.name == ref.name:
                continue
            print(fmt_compare(compare(base, r, ref, B=args.B, hi=args.hi, lo=args.lo), f"{r.name} − {ref.name}"))


if __name__ == "__main__":
    main()