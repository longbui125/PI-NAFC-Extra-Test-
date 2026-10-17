"""
build_anchors.py — Chọn câu anchor cho ràng buộc đúng → sai và đo mức nhiễu của hai cách đếm (chạy 1 lần).

Anchor = câu model gốc giải đúng ≥ 6/8 mẫu (p̂ ≥ 0,75), lấy từ phần train KHÔNG dùng để train: bỏ đúng
`--train_limit` câu mà prepare_buckets.py đã lấy (shuffle seed 0), bỏ câu trùng đề với tập test.

Mức nhiễu ghi vào meta (model gốc chấm lại trên chính các anchor, 4 mẫu, seed khác):
  noise_robust = tỷ lệ anchor có ≤ 1/4 mẫu đúng            → mức nhiễu của cách đếm "thật"  (NAFC, robust_fixed)
  noise_raw    = tỷ lệ anchor mẫu đầu lần chọn đúng mà mẫu đầu lần chấm lại sai → mức nhiễu của cách đếm "thô"
                                                                                      (FlipGuard, raw_adaptive)
Ví dụ:
  python build_anchors.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset math --train_limit 2000 --out anchors.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics

from common import chat_messages, is_correct, load_split


def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def select_pool(full: list[dict], used: list[dict], test: list[dict], pool: int, seed: int) -> tuple[list[dict], dict]:
    """Phần train chưa dùng, không trùng đề test, lấy ngẫu nhiên `pool` câu."""
    used_q = {norm_q(r["question"]) for r in used}
    test_q = {norm_q(r["question"]) for r in test}
    rest = [r for r in full if norm_q(r["question"]) not in used_q]
    clean = [r for r in rest if norm_q(r["question"]) not in test_q]
    rng = random.Random(seed + 1)
    chosen = rng.sample(clean, min(pool, len(clean)))
    return chosen, {"train_total": len(full), "train_used": len(used), "train_rest": len(rest),
                    "dup_with_test_removed": len(rest) - len(clean), "pool": len(chosen)}


def pick_anchors(scored: list[dict], n_anchor: int, hi: float, seed: int) -> list[dict]:
    cand = [x for x in scored if x["p_select"] >= hi and x["base_response"] is not None]
    rng = random.Random(seed)
    rng.shuffle(cand)
    return cand[:n_anchor]


def noise_floors(anchors: list[dict], lo: float) -> dict:
    n = max(len(anchors), 1)
    return {"noise_robust": sum(x["p_check"] <= lo for x in anchors) / n,
            "noise_raw": sum(x["base_first_correct"] and not x["check_first_correct"] for x in anchors) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math")
    ap.add_argument("--train_limit", type=int, default=2000, help="= LIMIT_TRAIN lúc tạo buckets train")
    ap.add_argument("--pool", type=int, default=800, help="số câu train chưa dùng đem chấm")
    ap.add_argument("--n_anchor", type=int, default=128)
    ap.add_argument("--k_select", type=int, default=8)
    ap.add_argument("--k_check", type=int, default=4)
    ap.add_argument("--hi", type=float, default=0.75)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0, help="= nhiệt độ lúc train và lúc đánh giá")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = load_split(args.dataset, "train", None)
    used = load_split(args.dataset, "train", args.train_limit)
    test = load_split(args.dataset, "test", None)
    pool, stats = select_pool(full, used, test, args.pool, args.seed)
    print("Chọn pool:", stats, flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len + 1024)

    def prompts(rows):
        return [tok.apply_chat_template(chat_messages(r["question"]), tokenize=False, add_generation_prompt=True)
                for r in rows]

    outs = llm.generate(prompts(pool), SamplingParams(n=args.k_select, temperature=args.temperature, top_p=1.0,
                                                      max_tokens=args.max_len, seed=args.seed))
    scored = []
    for r, o in zip(pool, outs):
        ok = [is_correct(c.text, r["answer"]) for c in o.outputs]
        good = [c for c, g in zip(o.outputs, ok) if g]
        scored.append({"question": r["question"], "answer": r["answer"], "p_select": sum(ok) / len(ok),
                       "base_first_correct": bool(ok[0]),
                       "base_response": good[0].text if good else None,
                       "base_response_ids": list(good[0].token_ids) if good else None,
                       "base_response_len": len(good[0].token_ids) if good else None})
    anchors = pick_anchors(scored, args.n_anchor, args.hi, args.seed)
    n_cand = sum(x["p_select"] >= args.hi for x in scored)
    if len(anchors) < args.n_anchor:
        print(f"CẢNH BÁO: chỉ có {len(anchors)} câu đạt p̂ ≥ {args.hi} trong pool {len(pool)}; tăng --pool nếu cần đủ "
              f"{args.n_anchor}.", flush=True)

    outs2 = llm.generate(prompts(anchors), SamplingParams(n=args.k_check, temperature=args.temperature, top_p=1.0,
                                                          max_tokens=args.max_len, seed=args.seed + 1))
    for x, o in zip(anchors, outs2):
        ok = [is_correct(c.text, x["answer"]) for c in o.outputs]
        x["p_check"] = sum(ok) / len(ok)
        x["check_first_correct"] = bool(ok[0])

    meta = {"model": args.model, "dataset": args.dataset, "train_limit": args.train_limit, **stats,
            "candidates": n_cand, "n_anchor": len(anchors), "k_select": args.k_select, "k_check": args.k_check,
            "hi": args.hi, "lo": args.lo, "max_len": args.max_len, "temperature": args.temperature,
            "pool_base_acc": statistics.mean(x["p_select"] for x in scored),
            "anchor_resp_len_mean": statistics.mean(x["base_response_len"] for x in anchors) if anchors else None,
            **noise_floors(anchors, args.lo)}
    json.dump({"meta": meta, "items": anchors}, open(args.out, "w"), ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import os as _os, sys as _sys
    main()
    _sys.stdout.flush(); _sys.stderr.flush()
    _os._exit(0)   # không chờ tiến trình nền của vLLM