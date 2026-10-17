"""
evaluate.py — Đánh giá một model (gốc hoặc đã train) theo đúng các metric của Proposal, mục 6.

Báo cáo: accuracy (pass@1, trung bình trên n mẫu), số token trung bình, và cả hai
THEO TỪNG NHÓM ĐỘ KHÓ, cùng tỷ lệ câu trả lời vượt ngân sách của nhóm.

Nhóm độ khó của tập test lấy từ file của prepare_buckets.py chạy trên split test
(nhóm được xác định bởi model GỐC, cố định cho mọi phương pháp).

Bản 14/10/2026: thêm trường "boxed" (tỷ lệ mẫu có \\boxed{}) cho từng câu, và tuỳ chọn --save_texts
để lưu nguyên văn lời giải (jsonl.gz) — dùng để kiểm tra bài "hỏng" là sai thật hay chỉ thiếu \\boxed{}.
Hai thay đổi này không đổi cách sinh hay cách chấm: acc, len, over, trunc tính y như cũ.

Ví dụ:
  python evaluate.py --model runs/dalg_s0/merged --buckets buckets_gsm8k_test.json --out results/dalg_s0.json
"""

from __future__ import annotations

import argparse
import json
import statistics

from common import chat_messages, is_correct, last_boxed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--buckets", required=True, help="file json của split test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0, help="= nhiệt độ khi train, để đo đúng phân phối đã ràng buộc")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1, help="KHÁC seed lúc chia nhóm (0): mẫu đánh giá độc lập với mẫu chia nhóm")
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_texts", default=None, help="đường dẫn .jsonl.gz: lưu lời giải từng câu (mặc định không lưu)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = json.load(open(args.buckets))
    meta, items = data["meta"], data["items"]
    max_len = args.max_len or meta["max_len"]
    budgets = meta["budgets"]
    K = len(budgets)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(chat_messages(it["question"]), tokenize=False, add_generation_prompt=True) for it in items]
    llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=max_len + 512)
    outs = llm.generate(prompts, SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                                                max_tokens=max_len, seed=args.seed))

    per_q = []
    all_ok = []
    for it, o in zip(items, outs):
        ok = [is_correct(c.text, it["answer"]) for c in o.outputs]
        all_ok.append(ok)
        ln = [len(c.token_ids) for c in o.outputs]
        per_q.append({"bucket": it["bucket"], "acc": sum(ok) / len(ok), "len": statistics.mean(ln),
                      "over": sum(L > budgets[it["bucket"]] for L in ln) / len(ln),
                      "trunc": sum(L >= max_len for L in ln) / len(ln),
                      "boxed": sum(last_boxed(c.text) is not None for c in o.outputs) / len(o.outputs)})

    if args.save_texts:
        import gzip
        with gzip.open(args.save_texts, "wt", encoding="utf-8") as f:
            for i, (it, o, ok) in enumerate(zip(items, outs, all_ok)):
                f.write(json.dumps({"i": i, "answer": it["answer"],
                                    "texts": [c.text for c in o.outputs],
                                    "ok": ok,
                                    "len": [len(c.token_ids) for c in o.outputs]}, ensure_ascii=False) + "\n")

    def summarize(rows):
        if not rows:
            return None
        return {"n": len(rows),
                "acc": round(100 * statistics.mean(r["acc"] for r in rows), 2),
                "mean_len": round(statistics.mean(r["len"] for r in rows), 1),
                "over_budget_rate": round(100 * statistics.mean(r["over"] for r in rows), 2),
                "truncated_rate": round(100 * statistics.mean(r["trunc"] for r in rows), 2),
                "boxed_rate": round(100 * statistics.mean(r["boxed"] for r in rows), 2)}

    result = {
        "model": args.model,
        "eval_set": f'{meta["dataset"]}/{meta["split"]}',
        "budget": meta["budget"],
        "decoding": {"temperature": args.temperature, "top_p": args.top_p, "n": args.n, "seed": args.seed},
        "budgets_by_bucket": budgets,
        "overall": summarize(per_q),
        "by_bucket": {k: summarize([r for r in per_q if r["bucket"] == k]) for k in range(K)},
        "per_question": per_q,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import os as _os, sys as _sys
    main()
    _sys.stdout.flush(); _sys.stderr.flush()
    _os._exit(0)   # thoát ngay: không chờ tiến trình nền của vLLM (từng làm ô treo hàng giờ)
