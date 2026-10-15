"""Repository PI/NAFC formulas; extracted unchanged from rl-nafc commit 3c18f8d."""

from __future__ import annotations

import math, json, re

from dataclasses import dataclass, field

def clipped_cost(length: int, budget: float) -> float:
    """Chi phí bị cắt của LACONIC: bằng 0 nếu trong ngân sách."""
    return max((length - budget) / budget, 0.0)

@dataclass
class LagrangianController:
    method: str
    budgets: list[float]            # B_k; với phương pháp toàn cục chỉ dùng budgets[0] = B
    max_len: int                    # L_max = max_completion_length
    difficulty: list[float] | None = None   # d_k trong [0,1], 0 = dễ, 1 = khó (dalg_param)
    eta: float = 0.002              # bước cập nhật đối ngẫu
    eta_param: float = 0.002        # bước cập nhật cho (a, b) của dalg_param
    fixed_lambda: float = 0.01
    optimistic: bool = False
    init_lambdas: list[float] | None = None
    leash_lambda_max: float = 1.0
    two_sided: bool = False         # λ được âm: ràng buộc coi là chủ động (E[L] = B), chống ép quá tay
    kp: float = 0.0                 # hệ số tỉ lệ (PI): λ = I + kp * vi_phạm, phản ứng ngay khi độ dài vượt/thiếu

    lambdas: list[float] = field(init=False)
    _prev_grad: list[float] = field(init=False)
    _a: float = field(init=False, default=0.0)
    _b: float = field(init=False, default=0.0)
    _prev_ab_grad: tuple[float, float] = field(init=False, default=(0.0, 0.0))
    _integral: list[float] = field(init=False)

    def __post_init__(self):
        valid = {"grpo", "fixed", "leash", "laconic", "dalg", "dalg_param"}
        if self.method not in valid:
            raise ValueError(f"method phải thuộc {valid}")
        k = self.num_groups
        if self.init_lambdas is not None:
            self.lambdas = [min(max(l, 0.0), self.ceiling(i)) for i, l in enumerate(self.init_lambdas[:k])]
        elif self.method == "fixed":
            self.lambdas = [self.fixed_lambda] * k
        else:
            self.lambdas = [0.0] * k
        self._prev_grad = [0.0] * k
        self._integral = list(self.lambdas)
        if self.method == "dalg_param":
            if self.difficulty is None or len(self.difficulty) != len(self.budgets):
                raise ValueError("dalg_param cần difficulty cùng độ dài với budgets")
            # Khởi tạo (a, b): đường thẳng qua lambda của nhóm dễ nhất và khó nhất (nếu có init), ngược lại (0, 0)
            if self.init_lambdas is not None:
                d0, d1 = self.difficulty[0], self.difficulty[-1]
                y0, y1 = self.lambdas[0], self.lambdas[-1]
                self._b = (y1 - y0) / (d1 - d0) if d1 != d0 else 0.0
                self._a = y0 - self._b * d0
            else:
                self._a, self._b = 0.0, 0.0
            self._sync_param_lambdas()

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        """Lưu trạng thái đối ngẫu để tiếp tục train khi Colab bị ngắt."""
        return {"lambdas": self.lambdas, "prev_grad": self._prev_grad,
                "a": self._a, "b": self._b, "prev_ab_grad": list(self._prev_ab_grad),
                "integral": list(self._integral)}

    def load_state_dict(self, d: dict) -> None:
        self.lambdas = list(d["lambdas"])
        self._prev_grad = list(d["prev_grad"])
        self._a, self._b = d["a"], d["b"]
        self._prev_ab_grad = tuple(d["prev_ab_grad"])
        self._integral = list(d.get("integral", self.lambdas))

    @property
    def num_groups(self) -> int:
        return len(self.budgets) if self.method in ("dalg", "dalg_param") else 1

    def group_of(self, bucket: int) -> int:
        return bucket if self.method in ("dalg", "dalg_param") else 0

    def ceiling(self, k: int) -> float:
        """Trần của lambda. LACONIC: Lambda = B/(Lmax - B) để lời giải đúng luôn có reward > 0."""
        if self.method == "leash":
            return self.leash_lambda_max
        b = self.budgets[k] if self.method in ("dalg", "dalg_param") else self.budgets[0]
        return b / max(self.max_len - b, 1.0)

    def floor(self, k: int) -> float:
        """Sàn của λ: 0 cho ràng buộc bất đẳng thức; -trần khi two_sided."""
        return -self.ceiling(k) if self.two_sided else 0.0

    def budget(self, k: int) -> float:
        return self.budgets[k] if self.method in ("dalg", "dalg_param") else self.budgets[0]

    def _sync_param_lambdas(self):
        self.lambdas = [
            min(max(self._a + self._b * d, 0.0), self.ceiling(i))
            for i, d in enumerate(self.difficulty)
        ]

    # ------------------------------------------------------------------ #
    def shaped_rewards(self, correct: list[bool], lengths: list[int], buckets: list[int]) -> list[float]:
        """Primal: reward Lagrange cho từng completion (dùng lambda hiện tại)."""
        out = []
        for ok, L, bk in zip(correct, lengths, buckets):
            k = self.group_of(bk)
            B = self.budget(k)
            lam = self.lambdas[k]
            if self.method == "grpo":
                out.append(1.0 if ok else 0.0)
            elif self.method == "leash":
                r = 1.0 if ok else -1.0
                out.append(min(max(r - lam * clipped_cost(L, B), -1.0), 1.0))
            else:  # fixed, laconic, dalg, dalg_param
                r = 1.0 if ok else 0.0
                out.append(r - lam * clipped_cost(L, B))
        return out

    def dual_update(self, lengths: list[int], buckets: list[int]) -> dict[str, float]:
        """Dual: cập nhật lambda từ độ dài trung bình (dùng chi phí tuyến tính L/B - 1, như LACONIC/Leash)."""
        stats: dict[str, float] = {}
        k_total = self.num_groups
        sums = [0.0] * k_total
        counts = [0] * k_total
        for L, bk in zip(lengths, buckets):
            k = self.group_of(bk)
            sums[k] += L
            counts[k] += 1

        grads = [0.0] * k_total
        for k in range(k_total):
            if counts[k] == 0:
                continue
            mean_len = sums[k] / counts[k]
            grads[k] = mean_len / self.budget(k) - 1.0
            stats[f"constraint/mean_len_g{k}"] = mean_len
            stats[f"constraint/violation_g{k}"] = grads[k]

        if self.method in ("grpo", "fixed"):
            pass
        elif self.method == "dalg_param":
            # Gradient ascent trên (a, b): dlambda_k/da = 1, dlambda_k/db = d_k (khi lambda_k không bị cắt).
            # Mỗi nhóm đóng góp theo tỷ lệ số mẫu -> (a, b) học từ TOÀN BỘ batch, ít nhiễu hơn K lambda độc lập.
            n = sum(counts) or 1
            ga = gb = 0.0
            for k in range(k_total):
                if counts[k] == 0:
                    continue
                w = counts[k] / n
                ga += w * grads[k]
                gb += w * grads[k] * self.difficulty[k]
            if self.optimistic:
                pa, pb = self._prev_ab_grad
                step_a, step_b = 2 * ga - pa, 2 * gb - pb
            else:
                step_a, step_b = ga, gb
            self._prev_ab_grad = (ga, gb)
            self._a += self.eta_param * step_a
            self._b += self.eta_param * step_b
            self._sync_param_lambdas()
            stats["constraint/param_a"] = self._a
            stats["constraint/param_b"] = self._b
        else:  # leash, laconic, dalg
            for k in range(k_total):
                if counts[k] == 0:
                    continue
                g = 2 * grads[k] - self._prev_grad[k] if self.optimistic else grads[k]
                self._prev_grad[k] = grads[k]
                lo, hi = self.floor(k), self.ceiling(k)
                if self.kp > 0:   # PI: phần tích phân (có chống tích luỹ quá trần) + phần tỉ lệ
                    self._integral[k] = min(max(self._integral[k] + self.eta * g, lo), hi)
                    self.lambdas[k] = min(max(self._integral[k] + self.kp * grads[k], lo), hi)
                else:
                    self.lambdas[k] = min(max(self.lambdas[k] + self.eta * g, lo), hi)

        for k, lam in enumerate(self.lambdas):
            stats[f"constraint/lambda_g{k}"] = lam
        return stats

MODES = ("nafc", "flipguard", "robust_fixed", "raw_adaptive", "monitor", "nafc_replay", "replay_fixed")

COUNTING = {"nafc": "robust", "flipguard": "raw", "robust_fixed": "robust", "raw_adaptive": "raw", "monitor": "robust",
            "nafc_replay": "robust", "replay_fixed": "robust"}

ADAPTIVE = {"nafc": True, "flipguard": False, "robust_fixed": False, "raw_adaptive": True, "monitor": False,
            "nafc_replay": True, "replay_fixed": False}

PULL = {"nafc": "nll", "flipguard": "nll", "robust_fixed": "nll", "raw_adaptive": "nll", "monitor": "none",
        "nafc_replay": "replay", "replay_fixed": "replay"}

def split_anchors(n: int, frac: float, seed: int = 0) -> tuple[list, list]:
    """Chia cố định n anchor: round(frac·n) anchor SỬA (được train), phần còn lại anchor ĐO (chỉ đo)."""
    import random
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    k = int(round(frac * n))
    return sorted(perm[:k]), sorted(perm[k:])

def n_replay_groups(lam: float, frac_max: float, n_groups: int, pool_size: int) -> int:
    """Số nhóm prompt thay bằng anchor: ceil(λ · frac_max · G), không vượt số anchor trong pool
    và luôn chừa ít nhất 1 prompt train."""
    if lam <= 0 or pool_size <= 0 or n_groups <= 1:
        return 0
    n = math.ceil(lam * frac_max * n_groups - 1e-9)
    return max(0, min(n, pool_size, n_groups - 1))

def replace_groups(inputs: list, n_replace: int, pool: list, ptr: int, make_item, key=None):
    """Thay n_replace nhóm prompt ĐẦU TIÊN của batch (dataset đã xáo) bằng anchor lấy lần lượt từ pool.
    Mọi phần tử cùng prompt (cùng nhóm GRPO) được thay bằng cùng một anchor.
    make_item(x, anchor_idx) -> phần tử mới. Trả về (inputs_mới, ptr_mới, danh_sách_anchor_đã_dùng)."""
    key = key or (lambda x: json.dumps(x["prompt"], sort_keys=True, ensure_ascii=False))
    order = []
    for x in inputs:
        k = key(x)
        if k not in order:
            order.append(k)
    chosen, used = {}, []
    for k in order[:n_replace]:
        a = pool[ptr % len(pool)]
        ptr += 1
        chosen[k] = a
        used.append(a)
    out = [make_item(x, chosen[key(x)]) if key(x) in chosen else x for x in inputs]
    return out, ptr, used

def robust_flags(correct: list[list[bool]], lo: float = 0.25) -> list[bool]:
    """Câu hỏng thật: tỷ lệ đúng của model hiện tại trên k mẫu ≤ lo (anchor đã chọn sẵn p^_gốc ≥ 0,75)."""
    return [len(c) > 0 and sum(c) / len(c) <= lo for c in correct]

def raw_flags(correct: list[list[bool]], base_first_correct: list[bool]) -> list[bool]:
    """Câu hỏng thô (kiểu FlipGuard): MỘT mẫu gốc đúng và MỘT mẫu mới sai (mẫu đầu tiên của mỗi bên)."""
    return [bool(b) and len(c) > 0 and not c[0] for c, b in zip(correct, base_first_correct)]

@dataclass
class FlipController:
    mode: str = "nafc"
    eps: float = 0.06            # ngân sách: tỷ lệ anchor bị hỏng cho phép (cùng cách đếm với chế độ)
    eta: float = 1.0             # bước đối ngẫu
    gamma: float = 0.1           # λ cố định cho flipguard / robust_fixed
    lam_max: float = 1.0
    lam: float = 0.0
    selected: list = field(default_factory=list)   # chỉ số anchor trong S (lần kiểm tra gần nhất)
    history: list = field(default_factory=list)
    repair_idx: list | None = None   # tách anchor: anchor SỬA (None = không tách, mọi anchor vừa sửa vừa đo)
    held_idx: list | None = None     # tách anchor: anchor ĐO, λ chạy theo F^ trên tập này

    @property
    def split(self) -> bool:
        return self.held_idx is not None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode phải thuộc {MODES}, nhận '{self.mode}'")
        if self.mode in ("flipguard", "robust_fixed", "replay_fixed"):
            self.lam = self.gamma
        if self.mode == "monitor":
            self.lam = 0.0

    @property
    def counting(self) -> str:
        return COUNTING[self.mode]

    @property
    def adaptive(self) -> bool:
        return ADAPTIVE[self.mode]

    @property
    def pull_kind(self) -> str:
        return PULL[self.mode]

    def replay_pool(self, n_anchor: int) -> list:
        """nafc_replay: anchor hỏng ở lần kiểm tra gần nhất (trong nửa SỬA nếu tách); replay_fixed: mọi anchor SỬA."""
        if self.mode == "replay_fixed":
            return list(self.repair_idx) if self.split else list(range(n_anchor))
        return list(self.selected)

    def update(self, robust: list[bool], raw: list[bool], step: int = -1, extra: dict | None = None) -> dict:
        """Một lần kiểm tra anchor: chọn S theo cách đếm của chế độ, cập nhật λ, trả về số liệu để ghi log."""
        n = max(len(robust), 1)
        f_robust, f_raw = sum(robust) / n, sum(raw) / n
        flags = robust if self.counting == "robust" else raw
        f = f_robust if self.counting == "robust" else f_raw
        self.selected = [i for i, x in enumerate(flags) if x]
        split_rec = {}
        if self.split:
            frac = lambda fl, idx: sum(1 for i in idx if fl[i]) / max(len(idx), 1)
            split_rec = {"F_robust_held": frac(robust, self.held_idx), "F_robust_repair": frac(robust, self.repair_idx),
                         "F_raw_held": frac(raw, self.held_idx), "F_raw_repair": frac(raw, self.repair_idx),
                         "n_held": len(self.held_idx), "n_repair": len(self.repair_idx)}
            f = frac(flags, self.held_idx)                                        # λ theo nửa ĐO
            rep_set = set(self.repair_idx)
            self.selected = [i for i in self.selected if i in rep_set]            # chỉ sửa nửa SỬA
        lam_before = self.lam
        if self.mode == "monitor":
            self.lam = 0.0
        elif self.adaptive:
            self.lam = min(max(self.lam + self.eta * (f - self.eps), 0.0), self.lam_max)
        else:
            self.lam = self.gamma
        rec = {"step": step, "mode": self.mode, "F_robust": f_robust, "F_raw": f_raw, "F_used": f, "eps": self.eps,
               "lam_before": lam_before, "lam": self.lam, "n_S": len(self.selected), "n_anchor": len(robust)}
        rec.update(split_rec)
        if extra:
            rec.update(extra)
        self.history.append(rec)
        return rec

    @property
    def pulling(self) -> bool:
        if self.mode == "replay_fixed":
            return self.lam > 0
        return self.lam > 0 and bool(self.selected)

    def state_dict(self) -> dict:
        return {"mode": self.mode, "eps": self.eps, "eta": self.eta, "gamma": self.gamma, "lam_max": self.lam_max,
                "lam": self.lam, "selected": list(self.selected), "history": list(self.history),
                "repair_idx": self.repair_idx, "held_idx": self.held_idx}

    def load_state_dict(self, d: dict) -> None:
        if d.get("mode") != self.mode:
            raise ValueError(f"checkpoint dùng chế độ {d.get('mode')}, lần chạy này là {self.mode}")
        if d.get("held_idx") != self.held_idx:
            raise ValueError("checkpoint dùng cách tách anchor khác lần chạy này (--anchor_split / --split_seed)")
        self.lam, self.selected, self.history = float(d["lam"]), list(d["selected"]), list(d["history"])

def default_eps(meta: dict, mode: str, margin: float = 0.02) -> float:
    """ε = mức nhiễu của CHÍNH cách đếm mà chế độ dùng (model gốc chấm lại trên anchor) + biên."""
    key = "noise_robust" if COUNTING[mode] == "robust" else "noise_raw"
    if key not in meta:
        raise KeyError(f"file anchor thiếu '{key}', chạy lại build_anchors.py")
    return float(meta[key]) + margin

def decoupled_advantages(group_keys, rewards, costs, lams, eps=1e-6):
    groups = {}
    for i, g in enumerate(group_keys):
        groups.setdefault(g, []).append(i)
    out = [0.0] * len(rewards)
    for idx in groups.values():
        def z(vals):
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return [0.0 if s < eps else (v - m) / s for v in vals]
        zr = z([rewards[i] for i in idx])
        zc = z([costs[i] for i in idx])
        for j, i in enumerate(idx):
            out[i] = zr[j] - lams[i] * zc[j]
    return out

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def gold_answer(dataset: str, raw: str) -> str:
    if dataset == "gsm8k":
        return raw.split("####")[-1].strip().replace(",", "")
    if dataset == "math":
        return (last_boxed(raw) or "").strip()
    return raw.strip()

def chat_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

def last_boxed(text: str) -> str | None:
    """Nội dung của \\boxed{...} cuối cùng, xử lý ngoặc lồng nhau."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None

def _normalize(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "").replace(",", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.rstrip(".")
    return s

def _as_number(s: str) -> float | None:
    s = _normalize(s)
    m = re.fullmatch(r"-?\d+(\.\d+)?", s)
    if m:
        return float(s)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m and int(m.group(3)) != 0:
        sign = -1 if m.group(1) else 1
        return sign * int(m.group(2)) / int(m.group(3))
    return None

def is_correct(completion: str, gold: str) -> bool:
    """Chấm đúng/sai. Ưu tiên math_verify nếu đã cài, ngược lại so sánh chuẩn hoá."""
    pred = last_boxed(completion)
    if pred is None:
        return False
    try:
        from math_verify import parse, verify

        g, p = parse(f"${gold}$"), parse(f"${pred}$")
        if g and p:
            return bool(verify(g, p))
    except Exception:
        pass
    a, b = _as_number(pred), _as_number(gold)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6 * max(1.0, abs(b))
    return _normalize(pred) == _normalize(gold)
