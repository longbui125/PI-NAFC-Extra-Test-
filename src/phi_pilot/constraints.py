"""
constraints.py — Bộ điều khiển nhân tử Lagrange cho GRPO có ràng buộc độ dài.

File này KHÔNG import torch/trl, nên có thể test độc lập (xem test_constraints.py).

Các phương pháp:
  - "grpo"       : GRPO thường, chỉ reward đúng/sai (không ràng buộc)
  - "fixed"      : penalty cố định, lambda không đổi (cost clipped như LACONIC)
  - "leash"      : Leash (Li et al., 2025) — reward {-1,+1}, penalty một phía, clip [-1,1],
                   dual update theo độ dài trung bình toàn cục
  - "laconic"    : LACONIC (Liu et al., 2026) — reward {0,1}, clipped cost,
                   dual update theo độ dài trung bình toàn cục, trần Lambda = B/(Lmax-B)
  - "dalg"       : ĐỀ XUẤT — K nhân tử độc lập, mỗi nhóm độ khó một ngân sách B_k
  - "dalg_param" : ĐỀ XUẤT — lambda_k = clip(a + b*d_k, 0, Lambda_k), chỉ 2 tham số đối ngẫu

Tuỳ chọn ổn định hoá (RQ3):
  - optimistic=True : dùng gradient dự đoán 2*g_t - g_{t-1} thay cho g_t
  - init_lambdas    : khởi tạo lambda từ tập hiệu chỉnh (xem calibrate_lambda)
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------- #
def calibrate_lambda(
    samples_by_prompt: list[list[tuple[bool, int]]],
    budget: float,
    lam_max: float,
    beta: float = 0.5,
    iters: int = 40,
) -> float:
    """
    Ước lượng lambda ban đầu cho MỘT nhóm từ mẫu của model gốc (chạy offline, trước khi train).

    Ý tưởng (theo hướng dual one-shot của CAN / LARA): policy tối ưu của bài toán
    KL-regularized là pi_lam(o|q) ∝ pi_0(o|q) * exp((r - lam*c)/beta).
    Dùng các mẫu o ~ pi_0 làm importance samples, ước lượng E_{pi_lam}[L] rồi
    tìm lambda nhỏ nhất sao cho E[L] <= budget bằng tìm kiếm nhị phân.

    samples_by_prompt: mỗi phần tử là danh sách (đúng/sai, độ dài) của các mẫu cho một prompt.
    Trả về lam_max nếu kể cả lam_max vẫn không thỏa ràng buộc.
    """

    def expected_len(lam: float) -> float:
        total, n = 0.0, 0
        for group in samples_by_prompt:
            if not group:
                continue
            logits = [((1.0 if ok else 0.0) - lam * clipped_cost(L, budget)) / beta for ok, L in group]
            m = max(logits)
            ws = [math.exp(x - m) for x in logits]
            z = sum(ws)
            total += sum(w * L for w, (_, L) in zip(ws, group)) / z
            n += 1
        return total / max(n, 1)

    if expected_len(0.0) <= budget:
        return 0.0
    if expected_len(lam_max) > budget:
        return lam_max
    lo, hi = 0.0, lam_max
    for _ in range(iters):
        mid = (lo + hi) / 2
        if expected_len(mid) > budget:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------- #
@dataclass
class AccuracyController:
    """
    Phương pháp đề xuất v4 ("acccon"): tối thiểu độ dài với ràng buộc accuracy theo nhóm độ khó

        min E[L]   s.t.   acc_k(t) >= tau_k(t)   cho mọi nhóm k,

    trong đó tau_k(t) = accuracy của GRPO (không ràng buộc) ở nhóm k tại CÙNG bước train t, trừ eps_k.
    Ngưỡng bám theo đường học của GRPO nên lambda không bị dồn lên trần ở giai đoạn đầu (khi mọi
    phương pháp đều còn kém), tránh kiểu phản ứng trễ đã gặp ở LACONIC.

    Lagrangian: max  -E[L] + sum_k lambda_k * acc_k  ->  với câu thuộc nhóm k, advantage (chuẩn hoá tách rời)
        A = (lambda_k * z(đúng) - z(L)) / (1 + lambda_k)
    Dual: lambda_k <- clip(lambda_k + eta * (tau_k(t) - acc_ema_k), lam_min, lam_max),
    acc_ema_k là trung bình trượt của accuracy nhóm k trong batch (mỗi bước chỉ có vài câu mỗi nhóm).
    Khác DALG: việc "nén lan" sang nhóm khác không tự thoả được ràng buộc accuracy — nếu nó làm nhóm
    vừa mất accuracy thì lambda của nhóm vừa TĂNG và kéo lại, chứ không về 0.
    """
    ref_curves: list[list[float]]     # tau_k(t) đã trừ eps, mỗi nhóm một danh sách theo bước
    eta: float = 0.5
    lam_init: float = 1.0
    lam_min: float = 0.5            # = mức ép mạnh nhất LACONIC từng dùng (λ≈2 tương đương 0,5 ở đây)
    lam_max: float = 4.0
    ema: float = 0.9

    lambdas: list[float] = field(init=False)
    acc_ema: list[float | None] = field(init=False)
    step: int = field(init=False, default=0)

    def __post_init__(self):
        k = len(self.ref_curves)
        self.lambdas = [min(max(self.lam_init, self.lam_min), self.lam_max)] * k
        self.acc_ema = [None] * k

    @property
    def num_groups(self) -> int:
        return len(self.ref_curves)

    def target(self, k: int) -> float:
        c = self.ref_curves[k]
        return c[min(self.step, len(c) - 1)]

    def state_dict(self) -> dict:
        return {"lambdas": self.lambdas, "acc_ema": self.acc_ema, "step": self.step}

    def load_state_dict(self, d: dict) -> None:
        self.lambdas, self.acc_ema, self.step = list(d["lambdas"]), list(d["acc_ema"]), int(d["step"])

    def dual_update(self, correct: list[bool], buckets: list[int]) -> dict[str, float]:
        stats: dict[str, float] = {}
        for k in range(self.num_groups):
            idx = [i for i, b in enumerate(buckets) if b == k]
            if not idx:
                continue
            a = sum(correct[i] for i in idx) / len(idx)
            self.acc_ema[k] = a if self.acc_ema[k] is None else self.ema * self.acc_ema[k] + (1 - self.ema) * a
            tau = self.target(k)
            self.lambdas[k] = min(max(self.lambdas[k] + self.eta * (tau - self.acc_ema[k]), self.lam_min), self.lam_max)
            stats[f"constraint/acc_ema_g{k}"] = self.acc_ema[k]
            stats[f"constraint/target_g{k}"] = tau
        for k, lam in enumerate(self.lambdas):
            stats[f"constraint/lambda_g{k}"] = lam
        self.step += 1
        return stats