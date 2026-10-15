"""One optimizer step per rollout batch; the repository's PI and NAFC on-policy rules."""
import os
import math
import time
from pathlib import Path
import torch
from core import LagrangianController, FlipController, decoupled_advantages, n_replay_groups, replace_groups, robust_flags, raw_flags
from data import schedule, evaluation_domains
from io_utils import OUT, read, write, digest
from model import load_model, Sampler, prompt_ids, token_logps, clipped_loss, evaluate

ARMS = ("grpo_pi", "grpo_pi_fixed_replay", "grpo_pi_nafc")
PAIRED_ARMS = ("grpo_pi", "grpo_pi_nafc")


def test_path(folder, step, final_step, domain):
    name = 'final' if step == final_step else f'step_{step}'
    return folder/f'{name}_{domain}_test.json'


def audit(sampler, bundle, ctrl, c, step):
    records = sampler(bundle["anchors"], 4, 500_000_000 + step * 1_000_000)
    correct = [[r["correct"] for r in records[i * 4:(i + 1) * 4]] for i in range(len(bundle["anchors"]))]
    p = [sum(x) / 4 for x in correct]
    flags = robust_flags(correct)
    raw = raw_flags(correct, [q["base_first_correct"] for q in bundle["anchors"]])
    return ctrl.update(flags, raw, step=step,
        extra={"anchor_p": p, "anchor_acc_held": sum(p[i] for i in bundle["held_idx"]) / len(bundle["held_idx"]),
               "anchor_acc_repair": sum(p[i] for i in bundle["repair_idx"]) / len(bundle["repair_idx"]),
               "capped": sum(r["capped"] for r in records) / len(records), "records": records})


def replay(rows, bundle, ctrl, c, ptr, arm="grpo_pi_nafc"):
    if arm == "grpo_pi_fixed_replay":
        pool = bundle["repair_idx"]
        n = min(math.ceil(c["fixed_replay_fraction"] * len(rows)), len(pool))
        return replace_groups(rows, n, pool, ptr, lambda old, i: bundle["anchors"][i], key=lambda x: x["id"])
    pool = ctrl.replay_pool(len(bundle["anchors"]))
    n = n_replay_groups(ctrl.lam, c["replay_frac_max"], len(rows), len(pool)) if ctrl.pulling else 0
    if n == 0:
        return rows, ptr, []
    return replace_groups(rows, n, pool, ptr, lambda old, i: bundle["anchors"][i], key=lambda x: x["id"])


def save_checkpoint(path, model, optimizer, scheduler, scaler, state):
    value = {"parameters": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "state": state}
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(saved["parameters"]) != expected:
        raise ValueError("Checkpoint trainable parameters differ.")
    with torch.no_grad():
        for n, p in saved["parameters"].items():
            params[n].copy_(p.to(params[n].device))
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    return saved["state"]


def train_arm(c, data, bundle, arm, loader=load_model, sampler_factory=Sampler):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "complete.json").exists():
        print(f"[{arm}] Complete; use a new outputs folder to start a different run.", flush=True)
        return
    if not (folder / "resume.pt").exists() and any(folder.iterdir()):
        raise ValueError(f"{folder} contains a partial run without a checkpoint; do not silently restart it.")
    model, tok = loader(c, c.get("reference_checkpoint", OUT / "reference_model"), trainable=True)
    model.train()
    sampler = sampler_factory(model, tok, c)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=c["learning_rate"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: min(t / c["warmup_steps"], 1.0))
    scaler = torch.cuda.amp.GradScaler(init_scale=c.get("loss_scale", 65536),
                                      enabled=c["precision"] == "fp16" and torch.cuda.is_available())
    pi = LagrangianController("laconic", budgets=[bundle["budget"]], max_len=c["max_new_tokens"],
                              eta=c["pi_eta"], kp=c["pi_kp"], two_sided=True)
    pi.ceiling = lambda k: c["pi_lambda_max"]
    nafc = FlipController(mode="nafc_replay" if arm == "grpo_pi_nafc" else "monitor", eps=bundle["eps"],
                          eta=c["nafc_eta"], lam_max=c["nafc_lambda_max"],
                          repair_idx=bundle["repair_idx"], held_idx=bundle["held_idx"])
    state = {"step": 0, "replay_ptr": 0, "last_audit_step": None, "metrics": [], "audits": [], "validation": {}}
    if (folder / "resume.pt").exists():
        state = load_checkpoint(folder / "resume.pt", model, optimizer, scheduler, scaler)
        pi.load_state_dict(state["pi"]); nafc.load_state_dict(state["nafc"])
        print(f"[{arm}] Resuming after batch {state['step']}.", flush=True)
    paired = schedule(data["task_train"], c["steps"], c["prompts_per_step"], c["seed"])
    def checkpoint():
        state.update(pi=pi.state_dict(), nafc=nafc.state_dict())
        save_checkpoint(folder / "resume.pt", model, optimizer, scheduler, scaler, state)
        write(folder / "train_metrics.json", state["metrics"])
        write(folder / "audit_metrics.json", state["audits"])
    def validation(step):
        for domain in ("numina",):
            key = f"{domain}_{step}"
            if key not in state["validation"]:
                records = evaluate(sampler, data[f"{domain}_validation"], 4, 600_000_000 + step * 1_000_000,
                                   folder / f"validation_{key}.json")
                state["validation"][key] = {"accuracy": sum(r["correct"] for r in records) / len(records),
                    "mean_tokens": sum(len(r["tokens"]) for r in records) / len(records)}
        checkpoint()
    def midpoint(step):
        # Save exact weights/state before a resumable audit so training cannot skip an interrupted milestone.
        checkpoint()
        model.save_pretrained(folder / 'adapters' / f'step_{step}')
        test_sampler = sampler_factory(model, tok, dict(c, do_sample=c.get('test_do_sample', True)))
        for domain in evaluation_domains(data):
            evaluate(test_sampler, data[f'{domain}_test'], c.get('test_draws', 4), 430_000_000,
                     test_path(folder, step, c['steps'], domain))
    midpoints = set(c.get('test_steps', ())) - {c['steps']}
    if not (folder / "resume.pt").exists():
        checkpoint()
    if state["step"] in c["validation_steps"]:
        validation(state["step"])
    if state['step'] in midpoints:
        midpoint(state['step'])
    for step in range(state["step"], c["steps"]):
        started = time.monotonic()
        rows, state["replay_ptr"], used = replay(paired[step], bundle, nafc, c, state["replay_ptr"], arm)
        records = sampler(rows, c["group_size"], 700_000_000 + step * 1_000_000 + c["seed"])
        expanded = [q for q in rows for _ in range(c["group_size"])]
        lengths = [len(r["tokens"]) for r in records]
        if len(records) != len(expanded) or any(r["id"] != q["id"] for r, q in zip(records, expanded)):
            raise ValueError("Incomplete or reordered task sampling")
        lam = pi.lambdas[0]
        adv = decoupled_advantages([q["id"] for q in expanded], [float(r["correct"]) for r in records],
                                   [(L - bundle["budget"]) / bundle["budget"] for L in lengths], [lam] * len(records))
        pi.dual_update(lengths, [0] * len(records))
        if step % c["audit_every"] == 0 and state["last_audit_step"] != step:
            full = audit(sampler, bundle, nafc, c, step)
            write(folder / f"audit_step_{step}.json", full)
            state["audits"].append({k: v for k, v in full.items() if k != "records"})
            nafc.history[-1].pop("records", None)
            state["last_audit_step"] = step
            print(f"[{arm}] audit {step}: held F={full['F_used']:.3%}, NAFC lambda={nafc.lam:.3f}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        active = False
        lr_used = optimizer.param_groups[0]["lr"]
        for q, record, advantage in zip(expanded, records, adv):
            if abs(advantage) < 1e-12:
                continue
            active = True
            logps = token_logps(model, prompt_ids(tok, q["question"]), record["tokens"])
            # One on-policy update: the detached current scores are the exact pre-update old-policy scores.
            loss = clipped_loss(logps, logps.detach(), advantage, c["max_new_tokens"]) / len(records)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite policy loss; last complete checkpoint is preserved.")
            scaler.scale(loss).backward()
            loss_total += float(loss.detach())
        if not active:
            for p in params:
                p.grad = torch.zeros_like(p)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not scaler.is_enabled() and not torch.isfinite(norm):
            raise FloatingPointError("Non-finite unscaled gradients")
        previous = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        updated = scaler.get_scale() >= previous
        if updated:
            scheduler.step()
        else:
            print(f"[{arm}] FP16 overflow: optimizer update skipped.", flush=True)
        state["step"] = step + 1
        state["metrics"].append({"step": step + 1, "loss": loss_total, "optimizer_updated": updated,
              "has_advantage_signal": active, "learning_rate": lr_used, "task_prompts": len(rows) - len(used),
              "anchor_prompts": len(used), "pi_lambda_used": lam, "pi_lambda_next": pi.lambdas[0],
              "nafc_lambda": nafc.lam, "accuracy": sum(r["correct"] for r in records) / len(records),
              "mean_tokens": sum(lengths) / len(lengths), "sampled_tokens": sum(lengths),
              "capped": sum(r["capped"] for r in records) / len(records), "seconds": time.monotonic() - started})
        print(f"[{arm}] {step + 1}/{c['steps']} acc={state['metrics'][-1]['accuracy']:.3f} "
              f"L={state['metrics'][-1]['mean_tokens']:.0f} replay={len(used)}", flush=True)
        if step + 1 in midpoints:
            midpoint(step + 1)
        elif (step + 1) % c.get("checkpoint_every", 25) == 0 or step + 1 == c["steps"]:
            checkpoint()
        if step + 1 in c["validation_steps"]:
            model.save_pretrained(folder / "adapters" / f"step_{step + 1}")
            validation(step + 1)
    model.save_pretrained(folder / "adapter_final")
    tok.save_pretrained(folder / "adapter_final")
    test_sampler = sampler_factory(model, tok, dict(c, do_sample=c.get("test_do_sample", True)))
    for domain in evaluation_domains(data):
        evaluate(test_sampler, data[f"{domain}_test"], c.get("test_draws", 4), 430_000_000,
                 test_path(folder, c['steps'], c['steps'], domain))
    if c.get("long_evaluation", True):
        long_sampler = sampler_factory(model, tok, dict(c, max_new_tokens=c["long_eval_tokens"]))
        for domain in evaluation_domains(data):
            evaluate(long_sampler, data[f"{domain}_test"], 4, 440_000_000,
                     folder / f"long_{domain}_test.json")
    write(folder / "complete.json", {"arm": arm, "step": state["step"], "validation": state["validation"],
          "test_steps": c.get('test_steps', [c['steps']]),
          "optimizer_updates": sum(x["optimizer_updated"] for x in state["metrics"]),
          "task_prompts": sum(x["task_prompts"] for x in state["metrics"]),
          "anchor_prompts": sum(x["anchor_prompts"] for x in state["metrics"]),
          "sampled_tokens": sum(x["sampled_tokens"] for x in state["metrics"]),
          "pi": pi.state_dict(), "nafc": nafc.state_dict()})
