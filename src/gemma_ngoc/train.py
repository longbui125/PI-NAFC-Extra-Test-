"""
train.py — GRPO có ràng buộc độ dài, trên 1 GPU (Colab A100), bằng TRL + vLLM (colocate) + LoRA.

Mọi phương pháp dùng CÙNG script, chỉ khác --method, để so sánh công bằng:
  grpo | fixed | leash | laconic | dalg | dalg_param

Cơ chế: TRL gọi reward function một lần cho mỗi batch sinh (generation batch).
Trong reward function ta
  (1) tính reward Lagrange với lambda hiện tại  (bước primal dùng reward này),
  (2) cập nhật lambda từ độ dài vừa quan sát     (bước dual),
nên mỗi bước tối ưu có đúng một lần cập nhật đối ngẫu, như LACONIC/Leash.

Ví dụ:
  python train.py --method laconic --buckets buckets_gsm8k_train.json --output_dir runs/laconic_s0 --seed 0
  python train.py --method dalg --init calibrated --buckets buckets_gsm8k_train.json --output_dir runs/dalg_s0

v8 (nhánh RL của đồ án gộp) thêm, không đổi gì khi để mặc định:
  --beta 0.04                         GRPO + KL với model gốc (TRL tắt LoRA để lấy model tham chiếu)
  --flip_mode nafc|flipguard|robust_fixed|raw_adaptive|monitor --anchors anchors.json
                                      ràng buộc đúng → sai (flipcon.py); monitor = chỉ đo, không kéo
v9 thêm --flip_mode nafc_replay|replay_fixed (sửa on-policy: anchor vào batch GRPO thay cho NLL) và
  --replay_frac_max (tỷ lệ prompt tối đa thay bằng anchor khi λ = 1)
v10 thêm --anchor_split 0.5 --split_seed 0: nửa anchor SỬA (được kéo/trộn), nửa ĐO (chỉ đo, điều khiển λ)
v11 thêm --keep_steps 50,100,150,200 (giữ adapter giữa chừng trong output_dir/keep/) và --save_total_limit
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from common import chat_messages, completion_text, is_correct
from constraints import AccuracyController, LagrangianController, clipped_cost


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


def build_controller(args, meta) -> LagrangianController:
    grouped = args.method in ("dalg", "dalg_param")
    budgets = meta["budgets"] if grouped else [meta["budget"]]
    init = None
    if args.lambda_init is not None and args.method not in ("grpo", "fixed"):
        init = [args.lambda_init] * (len(budgets))
    elif args.init == "calibrated" and args.method not in ("grpo", "fixed"):
        init = meta["init_lambdas"] if grouped else [meta["init_lambda_global"]]
    return LagrangianController(
        method=args.method,
        budgets=budgets,
        max_len=args.max_completion_length,
        difficulty=meta["difficulty"] if args.method == "dalg_param" else None,
        eta=args.eta,
        eta_param=args.eta,
        fixed_lambda=args.fixed_lambda,
        optimistic=args.optimistic,
        init_lambdas=init,
        leash_lambda_max=args.lambda_max,
        two_sided=args.two_sided,
        kp=args.kp,
    )


def acccon_advantages(group_keys, correct, lengths, lams, eps=1e-6):
    """A = (lam * z(đúng) - z_đúng(L)) / (1 + lam), chuẩn hoá trong từng nhóm rollout của cùng một prompt.
    z_đúng(L): độ dài chỉ được chuẩn hoá và phạt GIỮA CÁC LỜI GIẢI ĐÚNG (cần ít nhất 2); lời giải sai không
    nhận tín hiệu độ dài. Nhóm toàn sai -> advantage 0 (như GRPO), không bị đẩy viết ngắn."""
    groups = {}
    for i, g in enumerate(group_keys):
        groups.setdefault(g, []).append(i)
    out = [0.0] * len(correct)
    for idx in groups.values():
        def z(vals):
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return [0.0 if s < eps else (v - m) / s for v in vals]
        zc = z([1.0 if correct[i] else 0.0 for i in idx])
        ok = [i for i in idx if correct[i]]
        zl = dict(zip(ok, z([float(lengths[i]) for i in ok]))) if len(ok) >= 2 else {}
        for j, i in enumerate(idx):
            out[i] = (lams[i] * zc[j] - zl.get(i, 0.0)) / (1.0 + lams[i])
    return out


def make_acccon_reward_fn(ctrl: AccuracyController, num_buckets: int):
    def acccon_reward(completions, completion_ids, answer, bucket, log_metric=None, **kwargs):
        texts = [completion_text(c) for c in completions]
        correct = [is_correct(t, a) for t, a in zip(texts, answer)]
        lengths = [len(ids) for ids in completion_ids]
        buckets = [int(b) for b in bucket]
        lams = [ctrl.lambdas[b] for b in buckets]
        rewards = acccon_advantages([str(p) for p in kwargs["prompts"]], correct, lengths, lams)   # primal: lambda_t
        stats = ctrl.dual_update(correct, buckets)                                                # dual
        if log_metric is not None:
            log_metric("task/accuracy", sum(correct) / len(correct))
            log_metric("task/mean_len", statistics.mean(lengths))
            for k in range(num_buckets):
                idx = [i for i, b in enumerate(buckets) if b == k]
                if idx:
                    log_metric(f"task/accuracy_b{k}", sum(correct[i] for i in idx) / len(idx))
                    log_metric(f"task/mean_len_b{k}", statistics.mean(lengths[i] for i in idx))
            log_metric("constraint/mean_len_g0", statistics.mean(lengths))
            for name, v in stats.items():
                log_metric(name, v)
        return rewards

    return acccon_reward


def make_reward_fn(ctrl: LagrangianController, num_buckets: int, decoupled: bool = False, topq: float = 1.0):
    def lagrangian_reward(completions, completion_ids, answer, bucket, log_metric=None, **kwargs):
        texts = [completion_text(c) for c in completions]
        correct = [is_correct(t, a) for t, a in zip(texts, answer)]
        lengths = [len(ids) for ids in completion_ids]
        buckets = [int(b) for b in bucket]

        if decoupled:
            ks = [ctrl.group_of(b) for b in buckets]
            if ctrl.two_sided:   # chi phí có dấu: λ > 0 đẩy ngắn lại, λ < 0 kéo dài ra về phía B
                costs = [(L - ctrl.budget(k)) / ctrl.budget(k) for L, k in zip(lengths, ks)]
            else:
                costs = [clipped_cost(L, ctrl.budget(k)) for L, k in zip(lengths, ks)]
            lams = [ctrl.lambdas[k] for k in ks]
            keys = [str(p) for p in kwargs["prompts"]]
            if topq < 1.0:   # GRPO-λ: chỉ phạt độ dài ở các prompt có tỷ lệ đúng thuộc top q của batch
                acc_by = {}
                for i, kk in enumerate(keys):
                    acc_by.setdefault(kk, []).append(1.0 if correct[i] else 0.0)
                accs = {kk: sum(v) / len(v) for kk, v in acc_by.items()}
                ranked = sorted(accs.values(), reverse=True)
                n_keep = max(1, int(round(topq * len(ranked))))
                thr = ranked[n_keep - 1]
                lams = [l if accs[kk] >= thr and accs[kk] > 0 else 0.0 for l, kk in zip(lams, keys)]
            rewards = decoupled_advantages(keys, [1.0 if c else 0.0 for c in correct], costs, lams)
        else:
            rewards = ctrl.shaped_rewards(correct, lengths, buckets)   # primal: lambda_t
        stats = ctrl.dual_update(lengths, buckets)                 # dual: lambda_t -> lambda_{t+1}

        if log_metric is not None:
            log_metric("task/accuracy", sum(correct) / len(correct))
            log_metric("task/mean_len", statistics.mean(lengths))
            for k in range(num_buckets):
                idx = [i for i, b in enumerate(buckets) if b == k]
                if idx:
                    log_metric(f"task/accuracy_b{k}", sum(correct[i] for i in idx) / len(idx))
                    log_metric(f"task/mean_len_b{k}", statistics.mean(lengths[i] for i in idx))
            for name, v in stats.items():
                log_metric(name, v)
        return rewards

    return lagrangian_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["grpo", "fixed", "leash", "laconic", "dalg", "dalg_param", "acccon"])
    ap.add_argument("--buckets", required=True, help="file json từ prepare_buckets.py (split train)")
    ap.add_argument("--model", default=None, help="mặc định lấy từ file buckets")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    # Đối ngẫu
    ap.add_argument("--eta", type=float, default=0.002)
    ap.add_argument("--fixed_lambda", type=float, default=0.01)
    ap.add_argument("--init", choices=["zero", "calibrated"], default="zero")
    ap.add_argument("--lambda_init", type=float, default=None, help="khởi tạo lambda bằng hằng số (Leash dùng 0.1)")
    ap.add_argument("--lambda_max", type=float, default=1.0, help="trần lambda (leash; và mọi phương pháp khi --adv_norm decoupled)")
    ap.add_argument("--optimistic", action="store_true")
    ap.add_argument("--two_sided", action="store_true", help="λ được âm, chi phí có dấu (cần --adv_norm decoupled)")
    ap.add_argument("--kp", type=float, default=0.0, help="hệ số tỉ lệ của bộ điều khiển PI cho λ")
    ap.add_argument("--topq", type=float, default=1.0, help="GRPO-λ: chỉ phạt độ dài ở tỷ lệ topq prompt đúng nhiều nhất (1.0 = tắt)")
    ap.add_argument("--acc_ref", default=None, help="acccon: file json {\"curves\": [[tau_0(t)...], ...]} (đã trừ eps)")
    ap.add_argument("--acc_eta", type=float, default=0.5)
    ap.add_argument("--acc_lam_init", type=float, default=1.0)
    ap.add_argument("--acc_lam_min", type=float, default=0.5)
    ap.add_argument("--acc_lam_max", type=float, default=4.0)
    # GRPO
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--micro_batch", type=int, default=4, help="số completion mỗi lần forward/backward")
    ap.add_argument("--max_completion_length", type=int, default=None, help="mặc định = max_len trong file buckets")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--vllm_mem", type=float, default=0.3)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--save_total_limit", type=int, default=3, help="số checkpoint đầy đủ giữ lại (để train tiếp)")
    ap.add_argument("--keep_steps", default="", help="vd '50,100,150,200': chép adapter LoRA ở các bước này vào output_dir/keep/step-N "
                    "(chỉ adapter + trạng thái λ, không có optimizer) để đánh giá giữa chừng")
    ap.add_argument("--resume", default=None, help="đường dẫn checkpoint-xxx để train tiếp")
    ap.add_argument("--scale_rewards", default="none", choices=["group", "batch", "none"])
    ap.add_argument("--adv_norm", default="sum", choices=["sum", "decoupled"])
    ap.add_argument("--report_to", default="none", help="wandb | none")
    ap.add_argument("--no_vllm", action="store_true", help="chỉ để chạy thử trên CPU")
    ap.add_argument("--merged_dir", default=None, help="nơi lưu model đã gộp LoRA (mặc định output_dir/merged); nên để ngoài Drive vì ~3 GB")
    # v8: KL và ràng buộc đúng → sai
    ap.add_argument("--beta", type=float, default=0.0, help="hệ số KL với model gốc (0 = tắt, như mọi lần chạy trước)")
    ap.add_argument("--flip_mode", default="none", choices=["none", "nafc", "flipguard", "robust_fixed", "raw_adaptive", "monitor",
                                                           "nafc_replay", "replay_fixed"])
    ap.add_argument("--anchors", default=None, help="file json từ build_anchors.py")
    ap.add_argument("--anchor_limit", type=int, default=None, help="chỉ dùng N anchor đầu (chạy thử)")
    ap.add_argument("--flip_every", type=int, default=25, help="kiểm tra anchor mỗi N bước tối ưu")
    ap.add_argument("--flip_k", type=int, default=4, help="số mẫu mỗi anchor mỗi lần kiểm tra")
    ap.add_argument("--flip_eps", type=float, default=None, help="ngân sách; mặc định = mức nhiễu của cách đếm + flip_margin")
    ap.add_argument("--flip_margin", type=float, default=0.02)
    ap.add_argument("--flip_eta", type=float, default=1.0, help="bước đối ngẫu của λ (chế độ tự điều chỉnh)")
    ap.add_argument("--flip_gamma", type=float, default=0.1, help="λ cố định (flipguard, robust_fixed)")
    ap.add_argument("--flip_lam_max", type=float, default=1.0)
    ap.add_argument("--flip_per_micro", type=int, default=1, help="số anchor tính NLL mỗi micro-batch khi đang kéo")
    ap.add_argument("--anchor_split", type=float, default=0.0,
                    help="tỷ lệ anchor SỬA (được kéo/trộn vào train); phần còn lại chỉ để đo và điều khiển λ. 0 = không tách")
    ap.add_argument("--split_seed", type=int, default=0, help="seed của phép tách anchor (cố định giữa các lần chạy)")
    ap.add_argument("--replay_frac_max", type=float, default=0.5,
                    help="nafc_replay/replay_fixed: tỷ lệ prompt mỗi batch thay bằng anchor khi λ = 1 (n = ceil(λ·tỷ lệ·G))")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    data = json.load(open(args.buckets))
    meta = data["meta"]
    args.model = args.model or meta["model"]
    args.max_completion_length = args.max_completion_length or meta["max_len"]
    num_buckets = len(meta["budgets"])

    ds = Dataset.from_list([
        {"prompt": chat_messages(it["question"]), "answer": it["answer"], "bucket": it["bucket"]}
        for it in data["items"]
    ]).shuffle(seed=args.seed)

    if args.method == "acccon":
        assert args.acc_ref, "acccon cần --acc_ref"
        ctrl = AccuracyController(ref_curves=json.load(open(args.acc_ref))["curves"], eta=args.acc_eta,
                                  lam_init=args.acc_lam_init, lam_min=args.acc_lam_min, lam_max=args.acc_lam_max)
        args.scale_rewards, args.adv_norm = "none", "acccon"
    else:
        ctrl = build_controller(args, meta)
    if args.two_sided and args.adv_norm != "decoupled":
        ap.error("--two_sided cần --adv_norm decoupled")
    if args.method != "acccon" and args.adv_norm == "decoupled":
        args.scale_rewards = "none"
        ctrl.ceiling = lambda k: args.lambda_max
        ctrl.lambdas = [min(l, args.lambda_max) for l in ctrl.lambdas]
    if args.resume and os.path.exists(os.path.join(args.resume, "lambda_state.json")):
        ctrl.load_state_dict(json.load(open(os.path.join(args.resume, "lambda_state.json"))))
        print("Đã nạp lambda từ", args.resume, ctrl.lambdas)

    flip_ctrl, anchors = None, None
    if args.flip_mode != "none":
        from flipcon import AnchorBank, FlipController, default_eps, load_state, split_anchors
        assert args.anchors, "--flip_mode cần --anchors (file từ build_anchors.py)"
        anchors = AnchorBank.load(args.anchors, args.anchor_limit)
        eps = args.flip_eps if args.flip_eps is not None else default_eps(anchors.meta, args.flip_mode, args.flip_margin)
        repair_idx, held_idx = (split_anchors(len(anchors), args.anchor_split, args.split_seed)
                                if args.anchor_split > 0 else (None, None))
        flip_ctrl = FlipController(mode=args.flip_mode, eps=eps, eta=args.flip_eta, gamma=args.flip_gamma,
                                   lam_max=args.flip_lam_max, repair_idx=repair_idx, held_idx=held_idx)
        if held_idx is not None:
            print(f"Tách anchor: {len(repair_idx)} SỬA, {len(held_idx)} ĐO (split_seed={args.split_seed})", flush=True)
        if args.resume and load_state(flip_ctrl, args.resume):
            print("Đã nạp trạng thái ràng buộc đúng → sai từ", args.resume, "λ =", flip_ctrl.lam)
        print(f"Ràng buộc đúng → sai: mode={args.flip_mode}, {len(anchors)} anchor, ε={eps:.4f}, "
              f"mỗi {args.flip_every} bước × {args.flip_k} mẫu", flush=True)

    keep_steps = {int(x) for x in args.keep_steps.split(",") if x.strip()}
    bad = sorted(k for k in keep_steps if k % args.save_steps)
    if bad:
        ap.error(f"--keep_steps {bad} không chia hết cho --save_steps {args.save_steps}")

    class SaveLambda(TrainerCallback):
        def on_save(self, targs, state, control, **kw):
            path = os.path.join(targs.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(path, exist_ok=True)
            json.dump(ctrl.state_dict(), open(os.path.join(path, "lambda_state.json"), "w"))
            if flip_ctrl is not None:
                json.dump(flip_ctrl.state_dict(), open(os.path.join(path, "flip_state.json"), "w"))
            if state.global_step in keep_steps:
                import shutil
                dst = os.path.join(targs.output_dir, "keep", f"step-{state.global_step}")
                os.makedirs(dst, exist_ok=True)
                for f in ("adapter_model.safetensors", "adapter_config.json", "lambda_state.json", "flip_state.json"):
                    if os.path.exists(os.path.join(path, f)):
                        shutil.copy2(os.path.join(path, f), os.path.join(dst, f))
                print(f"Đã giữ adapter bước {state.global_step} → {dst}", flush=True)

    completions_per_step = args.prompts_per_step * args.num_generations
    assert completions_per_step % args.micro_batch == 0, "prompts_per_step*num_generations phải chia hết cho micro_batch"

    config = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=completions_per_step // args.micro_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=args.beta,
        loss_type="dr_grpo",        # bỏ chuẩn hoá theo độ dài câu trả lời, tránh nhiễu với ràng buộc độ dài
        scale_rewards=args.scale_rewards,
        use_vllm=not args.no_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        bf16=not args.no_vllm,
        gradient_checkpointing=not args.no_vllm,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        log_completions=False,
        model_init_kwargs={"dtype": "bfloat16"} if not args.no_vllm else None,
    )

    trainer_cls, extra = GRPOTrainer, {}
    if flip_ctrl is not None:
        from flipcon import make_flip_trainer
        trainer_cls = make_flip_trainer(GRPOTrainer)
        extra = dict(flip_ctrl=flip_ctrl, anchors=anchors, flip_every=args.flip_every, flip_k=args.flip_k,
                     flip_per_micro=args.flip_per_micro, replay_frac_max=args.replay_frac_max)

    trainer = trainer_cls(
        model=args.model,
        processing_class=AutoTokenizer.from_pretrained(args.model, padding_side="left"),
        reward_funcs=(make_acccon_reward_fn(ctrl, num_buckets) if args.method == "acccon"
                      else make_reward_fn(ctrl, num_buckets, args.adv_norm == "decoupled", args.topq)),
        args=config,
        train_dataset=ds,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[SaveLambda()],
        **extra,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()   # ghi trainer_state.json (log_history) để vẽ quỹ đạo lambda

    # Gộp LoRA để evaluate.py nạp thẳng bằng vLLM
    merged_dir = args.merged_dir or os.path.join(args.output_dir, "merged")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    trainer.processing_class.save_pretrained(merged_dir)
    info = {"method": args.method, "final_lambda_state": ctrl.state_dict(), "args": vars(args)}
    if flip_ctrl is not None:
        from flipcon import read_flip_log, summarize_log
        info["flip"] = {"final_state": flip_ctrl.state_dict(),
                        "summary": summarize_log(read_flip_log(os.path.join(args.output_dir, "flip_log.jsonl")))}
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    import os as _os, sys as _sys
    main()
    _sys.stdout.flush(); _sys.stderr.flush()
    _os._exit(0)   # thoát ngay: không chờ tiến trình nền của vLLM (từng làm ô treo hàng giờ)
