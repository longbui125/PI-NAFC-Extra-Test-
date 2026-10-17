"""
prepare_buckets.py — Bước offline trước khi train (Proposal, mục 5, bước 1–2).

Cho model gốc giải mỗi câu N lần bằng vLLM, rồi:
  1. tính tỷ lệ giải đúng p(q) và gán nhóm độ khó (dễ / vừa / khó),
  2. phân bổ ngân sách B_k theo độ dài trung vị lời giải đúng, co giãn để trung bình có trọng số = B,
  3. ước lượng lambda ban đầu cho từng nhóm (và cho phương pháp toàn cục).

Ví dụ:
  python prepare_buckets.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --dataset gsm8k --split train \
      --limit 2000 --budget 550 --out buckets_gsm8k_train.json

Chạy thêm một lần với --split test để có nhóm độ khó cho tập đánh giá
(dùng cùng --thresholds; ngân sách lấy từ file train, xem --budgets_from).
"""

from __future__ import annotations

import argparse
import json
import statistics

from common import chat_messages, is_correct, load_split
from constraints import calibrate_lambda


def bucket_of(p: float, thresholds: list[float]) -> int:
    """thresholds giảm dần, ví dụ [0.75, 0.25]: p>=0.75 -> 0 (dễ), p>=0.25 -> 1 (vừa), còn lại -> 2 (khó)."""
    for k, t in enumerate(thresholds):
        if p >= t:
            return k
    return len(thresholds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500", "math", "olympic"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=8, help="số mẫu mỗi câu")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--budget", type=float, default=550, help="ngân sách trung bình B")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.75, 0.25])
    ap.add_argument("--beta", type=float, default=0.5, help="nhiệt độ KL cho hiệu chỉnh lambda")
    ap.add_argument("--budgets_from", default=None, help="file json train để tái dùng B_k (cho tập test)")
    ap.add_argument("--reuse_samples", default=None, help="file json cũ: dùng lại mẫu, chỉ tính lại B_k và lambda (không cần GPU)")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.reuse_samples and not args.model:
        ap.error("cần --model (hoặc --reuse_samples)")
    if args.budgets_from:   # tập test: dùng CÙNG ngưỡng chia nhóm và ngân sách với tập train
        _ref = json.load(open(args.budgets_from))
        args.budget, args.thresholds, args.max_len = _ref["budget"], _ref["thresholds"], _ref["max_len"]

    if args.reuse_samples:
        old = json.load(open(args.reuse_samples))
        raw = [(it["question"], it["answer"], [tuple(x) for x in it["samples"]]) for it in old["items"]]
        args.model, args.max_len = old["meta"]["model"], old["meta"]["max_len"]
        args.dataset, args.split = old["meta"].get("dataset", args.dataset), old["meta"].get("split", args.split)
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        rows = load_split(args.dataset, args.split, args.limit)
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [tok.apply_chat_template(chat_messages(r["question"]), tokenize=False, add_generation_prompt=True) for r in rows]
        llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len + 512)
        sp = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_len, seed=0)
        outs = llm.generate(prompts, sp)
        raw = [(r["question"], r["answer"], [(is_correct(c.text, r["answer"]), len(c.token_ids)) for c in o.outputs])
               for r, o in zip(rows, outs)]

    K = len(args.thresholds) + 1
    items = []
    for q, a, samples in raw:
        p = sum(ok for ok, _ in samples) / len(samples)
        items.append({"question": q, "answer": a, "p": p, "bucket": bucket_of(p, args.thresholds), "samples": samples})

    counts = [sum(it["bucket"] == k for it in items) for k in range(K)]
    weights = [c / len(items) for c in counts]

    # Độ dài tự nhiên của model gốc theo nhóm (trung vị lời giải đúng; nhóm không có lời giải đúng -> trung vị mọi mẫu)
    natural = []
    for k in range(K):
        grp = [s for it in items if it["bucket"] == k for s in it["samples"]]
        good = [L for ok, L in grp if ok] or [L for _, L in grp] or [args.budget]
        natural.append(statistics.median(good))
    base_stats = {
        "acc_by_bucket": [statistics.mean(it["p"] for it in items if it["bucket"] == k) if counts[k] else None for k in range(K)],
        "mean_len_by_bucket": [statistics.mean(L for it in items if it["bucket"] == k for _, L in it["samples"]) if counts[k] else None for k in range(K)],
        "acc": statistics.mean(it["p"] for it in items),
        "mean_len": statistics.mean(L for it in items for _, L in it["samples"]),
    }

    if args.budgets_from:
        ref = json.load(open(args.budgets_from))
        budgets, init_group, init_global = ref["budgets"], ref["init_lambdas"], ref["init_lambda_global"]
    else:
        scale = args.budget / sum(w * n for w, n in zip(weights, natural))
        budgets = [n * scale for n in natural]
        init_group = []
        for k in range(K):
            grp = [it["samples"] for it in items if it["bucket"] == k]
            lam_max = budgets[k] / max(args.max_len - budgets[k], 1.0)
            init_group.append(calibrate_lambda(grp, budgets[k], lam_max, beta=args.beta) if grp else 0.0)
        lam_max_g = args.budget / max(args.max_len - args.budget, 1.0)
        init_global = calibrate_lambda([it["samples"] for it in items], args.budget, lam_max_g, beta=args.beta)

    meta = {
        "model": args.model, "dataset": args.dataset, "split": args.split, "n": args.n,
        "max_len": args.max_len, "budget": args.budget, "thresholds": args.thresholds,
        "counts": counts, "weights": weights, "natural_len": natural,
        "budgets": budgets, "difficulty": [k / (K - 1) for k in range(K)],
        "init_lambdas": init_group, "init_lambda_global": init_global,
        "base": base_stats,
    }
    with open(args.out, "w") as f:
        json.dump({"meta": meta, **meta, "items": items}, f, ensure_ascii=False)

    print(json.dumps({k: v for k, v in meta.items()}, indent=2, ensure_ascii=False))
    print("\nRQ1 — độ chính xác model gốc theo nhóm:", base_stats["acc_by_bucket"])


if __name__ == "__main__":
    import os as _os, sys as _sys
    main()
    _sys.stdout.flush(); _sys.stderr.flush()
    _os._exit(0)   # thoát ngay: không chờ tiến trình nền của vLLM (từng làm ô treo hàng giờ)
