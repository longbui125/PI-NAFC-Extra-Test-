"""
common.py — prompt, trích đáp án, chấm đúng/sai, nạp dữ liệu.

Dùng chung cho prepare_buckets.py, train.py, evaluate.py.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Tên dataset và cột. Kiểm tra lại trên Hugging Face trước khi chạy nếu đổi dataset.
DATASETS = {
    "gsm8k": dict(path="openai/gsm8k", name="main", question="question", answer="answer"),
    "math500": dict(path="HuggingFaceH4/MATH-500", name=None, question="problem", answer="answer"),
    "math": dict(path="DigitalLearningGmbH/MATH-lighteval", name=None, question="problem", answer="solution"),
}


def gold_answer(dataset: str, raw: str) -> str:
    if dataset == "gsm8k":
        return raw.split("####")[-1].strip().replace(",", "")
    if dataset == "math":
        return (last_boxed(raw) or "").strip()
    return raw.strip()


def load_split(dataset: str, split: str, limit: int | None = None):
    """Trả về list dict {question, answer} đã chuẩn hoá."""
    from datasets import load_dataset

    cfg = DATASETS[dataset]
    ds = load_dataset(cfg["path"], cfg["name"], split=split) if cfg["name"] else load_dataset(cfg["path"], split=split)
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=0).select(range(limit))   # lấy ngẫu nhiên, tránh dồn vào một chủ đề
    rows = [{"question": r[cfg["question"]], "answer": gold_answer(dataset, r[cfg["answer"]])} for r in ds]
    return [r for r in rows if r["answer"]]


def chat_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


# ---------------------------------------------------------------------- #
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


def completion_text(c) -> str:
    """TRL truyền completion dạng chuỗi (standard) hoặc list message (conversational)."""
    if isinstance(c, list):
        return c[-1]["content"] if c else ""
    return c