"""
flipcon.py — Ràng buộc đúng → sai (chống làm hỏng câu model gốc đã giải được) cho GRPO trên TRL.

Một lớp, năm chế độ; hai "công tắc" của ablation 2×2 là cách đếm câu hỏng và cách đặt λ:

  mode           đếm câu hỏng                 λ                                   vai trò
  ------------   --------------------------   ---------------------------------   -------------------------------
  nafc           thật: p^_θ ≤ 0,25 trên k mẫu  tự điều chỉnh theo ngân sách ε      phương pháp đề xuất (NAFC)
  flipguard      thô: 1 mẫu gốc đúng, 1 mẫu   γ cố định                           FlipGuard (Zhu và cs., 2024),
                 mới sai                                                           bản cho bài có đáp án kiểm được
  robust_fixed   thật                          γ cố định                           ô ablation
  raw_adaptive   thô                           tự điều chỉnh theo ngân sách ε      ô ablation
  monitor        thật                          luôn 0 (chỉ đo, không kéo)          đường đối chứng cho GRPO / KL
  nafc_replay    thật                          tự điều chỉnh theo ngân sách ε      NAFC on-policy: sửa bằng cách đưa anchor
                                                                                   hỏng vào batch GRPO (không dùng NLL)
  replay_fixed   thật (chỉ đo)                 γ cố định                           đối chứng: luôn trộn anchor ngẫu nhiên
                                                                                   (rehearsal thường, không có bộ phát hiện)

Cách hoạt động (mỗi `every` bước tối ưu, ngay sau khi TRL sinh lời giải của bước đó):
  1. Sinh k mẫu cho mỗi câu anchor bằng model hiện tại (vLLM colocate của TRL; không có vLLM thì HF generate).
  2. Đếm câu hỏng: F^ = tỷ lệ anchor bị hỏng theo cách đếm của chế độ; S = tập anchor đó.
  3. Cập nhật λ: tự điều chỉnh  λ ← clip(λ + η(F^ − ε), 0, λ_max);  cố định  λ = γ;  monitor  λ = 0.
  4. Ở mọi micro-batch cho đến lần kiểm tra sau: loss = loss GRPO + λ · NLL(lời giải đúng của model gốc | câu ∈ S),
     chia cho số bước tích luỹ gradient như loss GRPO (không thêm bước tối ưu nào, mọi nhánh cùng số bước).
  4'. Chế độ replay (nafc_replay, replay_fixed) KHÔNG dùng NLL. Ở mỗi lần sinh cho đến lần kiểm tra sau, thay
     n = ceil(λ · replay_frac_max · số prompt) nhóm prompt của batch bằng anchor (nafc_replay: lấy từ S;
     replay_fixed: lấy lần lượt từ mọi anchor). Anchor được sinh 8 lời giải, chấm và nhận advantage GRPO như mọi
     prompt khác, nên chịu đúng phần thưởng của phương pháp chính (kể cả phạt độ dài của LACONIC).
     Tổng số lời giải mỗi bước giữ nguyên, nên mọi nhánh cùng ngân sách tính toán.
  Tách anchor (v10, --anchor_split): chia cố định anchor thành nửa SỬA (được kéo / trộn vào train) và nửa ĐO
     (không bao giờ được train). λ cập nhật theo F^ trên nửa ĐO; S chỉ lấy anchor hỏng trong nửa SỬA. Nhờ vậy
     tín hiệu điều khiển và số liệu theo dõi không bị thổi phồng vì model đã học lại chính các anchor đó.

Anchor (từ build_anchors.py) là những câu model gốc đúng ≥ 6/8 mẫu, lấy từ phần train KHÔNG dùng để train;
file có sẵn mức nhiễu của cả hai cách đếm (model gốc chấm lại), nên mặc định ε = mức nhiễu + 2 điểm %.

Phần thuần Python (FlipController, AnchorBank.load, flags) kiểm thử không cần GPU: test_flipcon.py.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

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


# --------------------------------------------------------------------------------------------------------------- #
# Đếm câu hỏng
# --------------------------------------------------------------------------------------------------------------- #
def robust_flags(correct: list[list[bool]], lo: float = 0.25) -> list[bool]:
    """Câu hỏng thật: tỷ lệ đúng của model hiện tại trên k mẫu ≤ lo (anchor đã chọn sẵn p^_gốc ≥ 0,75)."""
    return [len(c) > 0 and sum(c) / len(c) <= lo for c in correct]


def raw_flags(correct: list[list[bool]], base_first_correct: list[bool]) -> list[bool]:
    """Câu hỏng thô (kiểu FlipGuard): MỘT mẫu gốc đúng và MỘT mẫu mới sai (mẫu đầu tiên của mỗi bên)."""
    return [bool(b) and len(c) > 0 and not c[0] for c, b in zip(correct, base_first_correct)]


# --------------------------------------------------------------------------------------------------------------- #
# Bộ điều khiển λ
# --------------------------------------------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------------------------------------------- #
# Anchor
# --------------------------------------------------------------------------------------------------------------- #
@dataclass
class AnchorBank:
    questions: list
    answers: list
    base_responses: list           # một lời giải đúng của model gốc cho mỗi câu (text)
    base_response_ids: list        # token id tương ứng (nếu có), ưu tiên dùng khi tính NLL
    base_first_correct: list       # mẫu đầu tiên của model gốc có đúng không (cho cách đếm thô)
    meta: dict

    @classmethod
    def load(cls, path: str, limit: int | None = None) -> "AnchorBank":
        d = json.load(open(path))
        items = d["items"][:limit] if limit else d["items"]
        if not items:
            raise ValueError(f"{path} không có anchor nào")
        return cls(questions=[x["question"] for x in items], answers=[x["answer"] for x in items],
                   base_responses=[x["base_response"] for x in items],
                   base_response_ids=[x.get("base_response_ids") for x in items],
                   base_first_correct=[bool(x["base_first_correct"]) for x in items], meta=d["meta"])

    def __len__(self):
        return len(self.questions)


# --------------------------------------------------------------------------------------------------------------- #
# Tích hợp vào GRPOTrainer của TRL
# --------------------------------------------------------------------------------------------------------------- #
def make_flip_trainer(base_cls):
    """Trả về lớp con của GRPOTrainer (TRL 1.13) có thêm ràng buộc đúng → sai. Import torch/TRL muộn."""
    import torch
    import torch.nn.functional as F

    from common import chat_messages, is_correct

    class FlipGRPOTrainer(base_cls):
        def __init__(self, *args, flip_ctrl: FlipController, anchors: AnchorBank, flip_every: int = 25,
                     flip_k: int = 4, flip_per_micro: int = 1, replay_frac_max: float = 0.5, scorer=None,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.flip_ctrl, self.anchors = flip_ctrl, anchors
            self.flip_every, self.flip_k, self.flip_per_micro = flip_every, flip_k, flip_per_micro
            self.flip_scorer = scorer or is_correct
            self._flip_last_check = None
            self._flip_ptr = 0
            self.replay_frac_max = replay_frac_max
            self._replay_ptr = 0
            self._anchor_msgs = [chat_messages(q) for q in anchors.questions]
            tok = self.processing_class
            self._anchor_prompts = [tok.apply_chat_template(chat_messages(q), tokenize=False, add_generation_prompt=True)
                                    for q in anchors.questions]
            self._anchor_prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in self._anchor_prompts]
            eos = tok.eos_token_id
            self._anchor_resp_ids = []
            for text, ids in zip(anchors.base_responses, anchors.base_response_ids):
                r = list(ids) if ids else tok(text, add_special_tokens=False)["input_ids"]
                if eos is not None and (not r or r[-1] != eos):
                    r = r + [eos]
                self._anchor_resp_ids.append(r[: self.max_completion_length])
            self.flip_log_path = os.path.join(self.args.output_dir, "flip_log.jsonl")

        # ---------------- sinh mẫu cho anchor ----------------
        def _anchor_samples(self) -> list[list[str]]:
            k = self.flip_k
            if getattr(self, "use_vllm", False) and getattr(self.vllm_generation, "mode", None) == "colocate":
                from vllm import SamplingParams
                if self.state.global_step != self._last_loaded_step:   # TRL thường đã đồng bộ trong bước sinh
                    self.vllm_generation.sync_weights()
                    self._last_loaded_step = self.state.global_step
                sp = SamplingParams(n=k, temperature=self.temperature, top_p=1.0, max_tokens=self.max_completion_length)
                outs = self.vllm_generation.llm.generate(self._anchor_prompts, sp, use_tqdm=False)
                return [[c.text for c in o.outputs] for o in outs]
            # Đường dự phòng (CPU / không vLLM): HF generate từng câu
            from trl.models import unwrap_model_for_generation
            tok = self.processing_class
            texts = []
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as m, torch.no_grad():
                was_training = m.training
                m.eval()
                for ids in self._anchor_prompt_ids:
                    x = torch.tensor([ids], device=self.accelerator.device)
                    out = m.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True,
                                     temperature=self.temperature, top_p=1.0, num_return_sequences=k,
                                     max_new_tokens=self.max_completion_length, pad_token_id=tok.pad_token_id)
                    texts.append(tok.batch_decode(out[:, x.size(1):], skip_special_tokens=True))
                if was_training:
                    m.train()
            return texts

        def _maybe_check_anchors(self):
            step = self.state.global_step
            if step % self.flip_every != 0 or step == self._flip_last_check:
                return
            self._flip_last_check = step
            t0 = time.time()
            samples = self._anchor_samples()
            correct = [[bool(self.flip_scorer(t, a)) for t in ts] for ts, a in zip(samples, self.anchors.answers)]
            robust = robust_flags(correct)
            raw = raw_flags(correct, self.anchors.base_first_correct)
            p = [sum(c) / max(len(c), 1) for c in correct]
            acc = sum(p) / max(len(p), 1)
            extra = {"anchor_acc": acc, "check_seconds": round(time.time() - t0, 1), "anchor_p": [round(v, 3) for v in p]}
            ctrl = self.flip_ctrl
            if ctrl.split:
                mean_on = lambda idx: sum(p[i] for i in idx) / max(len(idx), 1)
                extra.update(anchor_acc_held=mean_on(ctrl.held_idx), anchor_acc_repair=mean_on(ctrl.repair_idx))
            rec = ctrl.update(robust, raw, step=step, extra=extra)
            keys = ["F_robust", "F_raw", "lam", "n_S", "anchor_acc"]
            if ctrl.split:
                keys += ["F_robust_held", "F_robust_repair", "anchor_acc_held", "anchor_acc_repair"]
            for key in keys:
                self._metrics["train"][f"flip/{key}"].append(float(rec[key]))
            if self.accelerator.is_main_process:
                with open(self.flip_log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                tail = (f" | ĐO: F={rec['F_robust_held']:.3f} acc={rec['anchor_acc_held']:.3f}"
                        f" | SỬA: F={rec['F_robust_repair']:.3f} acc={rec['anchor_acc_repair']:.3f}") if ctrl.split else ""
                print(f"[flip] bước {step}: F_robust={rec['F_robust']:.3f} F_raw={rec['F_raw']:.3f} "
                      f"λ={rec['lam']:.3f} |S|={rec['n_S']} acc_anchor={acc:.3f}{tail} ({rec['check_seconds']}s)", flush=True)

        # ---------------- replay on-policy: đưa anchor vào batch GRPO ----------------
        def _replay(self, inputs):
            ctrl = self.flip_ctrl
            pool = ctrl.replay_pool(len(self.anchors))
            keyf = lambda x: json.dumps(x["prompt"], sort_keys=True, ensure_ascii=False)
            n_groups = len({keyf(x) for x in inputs})
            n = n_replay_groups(ctrl.lam, self.replay_frac_max, n_groups, len(pool)) if ctrl.pulling else 0
            if n == 0:
                self._metrics["train"]["flip/n_replay"].append(0.0)
                return inputs

            def make_item(x, a):
                y = dict(x)
                y["prompt"] = self._anchor_msgs[a]
                y["answer"] = self.anchors.answers[a]
                if "bucket" in y:
                    y["bucket"] = 0          # anchor là bài model gốc đúng >= 6/8 lần: nhóm dễ
                return y

            out, self._replay_ptr, used = replace_groups(inputs, n, pool, self._replay_ptr, make_item, key=keyf)
            self._metrics["train"]["flip/n_replay"].append(float(len(used)))
            return out

        def _generate_and_score_completions(self, inputs):
            if self.model.training and self.flip_ctrl.pull_kind == "replay":
                inputs = self._replay(inputs)
            out = super()._generate_and_score_completions(inputs)
            if self.model.training:
                self._maybe_check_anchors()
            return out

        # ---------------- loss học lại lời giải gốc trên S ----------------
        def _anchor_nll(self, model, idx: int):
            p, r = self._anchor_prompt_ids[idx], self._anchor_resp_ids[idx]
            dev = self.accelerator.device
            x = torch.tensor([p + r], device=dev)
            kw = {"input_ids": x, "attention_mask": torch.ones_like(x), "use_cache": False}
            if "logits_to_keep" in getattr(self, "model_kwarg_keys", ()):
                kw["logits_to_keep"] = len(r) + 1
            logits = model(**kw).logits[:, -(len(r) + 1):-1, :]
            target = torch.tensor([r], device=dev)
            return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1))

        def _compute_loss(self, model, inputs):
            loss = super()._compute_loss(model, inputs)
            if not self.model.training or not self.flip_ctrl.pulling or self.flip_ctrl.pull_kind != "nll":
                return loss
            S = self.flip_ctrl.selected
            nll = 0.0
            for _ in range(self.flip_per_micro):
                nll = nll + self._anchor_nll(model, S[self._flip_ptr % len(S)])
                self._flip_ptr += 1
            nll = nll / self.flip_per_micro
            self._metrics["train"]["flip/anchor_nll"].append(float(nll.detach()))
            return loss + self.flip_ctrl.lam * nll / self.current_gradient_accumulation_steps

    return FlipGRPOTrainer


def save_state(ctrl: FlipController, ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    json.dump(ctrl.state_dict(), open(os.path.join(ckpt_dir, "flip_state.json"), "w"))


def load_state(ctrl: FlipController, ckpt_dir: str) -> bool:
    p = os.path.join(ckpt_dir, "flip_state.json")
    if not os.path.exists(p):
        return False
    ctrl.load_state_dict(json.load(open(p)))
    return True


def read_flip_log(path: str) -> list[dict]:
    """Đọc flip_log.jsonl; nếu train tiếp từ checkpoint thì một bước có thể bị ghi hai lần → giữ bản sau cùng."""
    if not os.path.exists(path):
        return []
    by_step = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def summarize_log(recs: list[dict]) -> dict:
    if not recs:
        return {}
    lam = [r["lam"] for r in recs]
    return {"checks": len(recs), "lam_max_seen": max(lam), "lam_mean": sum(lam) / len(lam),
            "F_robust_mean": sum(r["F_robust"] for r in recs) / len(recs),
            "F_raw_mean": sum(r["F_raw"] for r in recs) / len(recs),
            "frac_checks_pulling": sum(r["lam"] > 0 and r["n_S"] > 0 for r in recs) / len(recs),
            "check_seconds_mean": sum(r.get("check_seconds", math.nan) for r in recs) / len(recs)}
