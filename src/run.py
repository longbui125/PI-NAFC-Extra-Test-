"""Shared Phi reference, then three paired arms for every prespecified seed."""
import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "hf_cache"))
from core import robust_flags, split_anchors
from data import prepare_splits, validate_splits, describe, evaluation_domains
from io_utils import ROOT, OUT, read, write, freeze, digest, file_hash, checkpoint_identity, validate_config, experiment_identity
from model import torch, check_gpu, load_model, base_path, Sampler, evaluate, prompt_ids, clear_memory

SHARED = ROOT / "outputs_phi_olympic" / "shared"


def set_output(seed=None):
    global OUT
    OUT = SHARED if seed is None else ROOT / "outputs_phi_olympic" / f"seed_{seed}"
    for name in ("io_utils", "model", "engine", "report"):
        if name in sys.modules:
            sys.modules[name].OUT = OUT
    return OUT


def environment(c):
    return {"versions": {k: importlib.metadata.version(k) for k in
            ("torch", "transformers", "peft", "datasets", "accelerate", "math-verify", "numpy", "matplotlib")},
            "gpu": torch.cuda.get_device_name(0), "precision": c["precision"]}


def identity(c, parent, splits):
    if file_hash(ROOT / "core.py") != read(ROOT / "SOURCE.json")["core_sha256"]:
        raise ValueError("The upstream PI/NAFC formulas changed.")
    return experiment_identity(c, parent, splits, environment(c))


def eligible(selection, c):
    return [q for q, rr in selection if len(rr) == c["initial_draws"]
            and sum(r["correct"] for r in rr) / len(rr) >= .75]


def prepare_anchors(c, data, sampler):
    path = OUT / "anchor_selection.json"
    state = read(path) if path.exists() else {"tested": 0, "records": []}
    pool = data["anchor_candidates"]
    n = c["initial_draws"]
    if [(r["id"], r["sample_index"]) for r in state["records"]] != [
            (q["id"], j) for q in pool[:state["tested"]] for j in range(n)]:
        raise ValueError("Anchor selection progress does not match frozen candidates.")
    def grouped():
        return [(q, state["records"][i*n:(i+1)*n]) for i, q in enumerate(pool[:state["tested"]])]
    candidates = eligible(grouped(), c)
    while len(candidates) < c["anchor_count"]:
        start = state["tested"]
        batch = pool[start:start + c["anchor_selection_batch"]]
        if not batch:
            raise ValueError(f"Stable anchors {len(candidates)}/{c['anchor_count']}. Frozen pool exhausted; no training started.")
        rr = sampler(batch, n, 300_000_000 + start*n*1009)
        if [(r["id"], r["sample_index"]) for r in rr] != [(q["id"], j) for q in batch for j in range(n)]:
            raise ValueError("Incomplete anchor sampling")
        state["records"].extend(rr); state["tested"] += len(batch)
        write(path, state)
        candidates = eligible(grouped(), c)
        print(f"[anchors] {len(candidates)}/{c['anchor_count']} stable; tested {state['tested']}", flush=True)
    import random
    random.Random(c["split_seed"]).shuffle(candidates)
    first = {q["id"]: rr[0]["correct"] for q, rr in grouped()}
    anchors = [dict(q, base_first_correct=first[q["id"]]) for q in candidates[:c["anchor_count"]]]
    repair, held = split_anchors(len(anchors), .5, 0)
    noise = []
    for repeat in range(3):
        rr = evaluate(sampler, [anchors[i] for i in held], 4, 400_000_000 + repeat*1_000_000,
                      OUT / f"anchor_noise_{repeat}.json")
        flags = robust_flags([[r["correct"] for r in rr[i*4:(i+1)*4]] for i in range(len(held))])
        noise.append(sum(flags) / len(held))
    return {"anchors": anchors, "repair_idx": repair, "held_idx": held, "noise_repeats": noise,
            "noise_robust": sum(noise)/3, "eps": sum(noise)/3 + c["noise_margin"],
            "selection_tested": state["tested"]}


def prepare(c):
    set_output()
    check_gpu(c)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / "config.json", c)
    data = read(OUT / "splits.json") if (OUT / "splits.json").exists() else prepare_splits(c)
    validate_splits(data, c); freeze(OUT / "splits.json", data)
    print("[model] Downloading/loading pinned Phi reference", flush=True)
    path = base_path(c)
    freeze(OUT / "identity.json", identity(c, checkpoint_identity(path), data))
    if (OUT / "prepared.json").exists():
        print("[prepare] Shared preparation is complete.", flush=True)
        return
    print(describe(data), flush=True)
    model, tok = load_model(c, path)
    for name, rows in data.items():
        if name != "provenance" and any(len(prompt_ids(tok, q["question"])) > c["max_prompt_tokens"] for q in rows):
            raise ValueError(f"Prompt exceeds cap in {name}; no silent truncation or row removal.")
    sampler = Sampler(model, tok, c)
    bundle = prepare_anchors(c, data, sampler)
    val = evaluate(sampler, data["numina_validation"], 4, 410_000_000, OUT / "reference_validation.json")
    mean = sum(len(r["tokens"]) for r in val) / len(val)
    bundle.update(budget=max(1, round(mean*c["compression_ratio"])), calibration_mean_tokens=mean,
                  budget_rule="70% of frozen Phi reference validation mean; independent of test results")
    long_sampler = Sampler(model, tok, dict(c, max_new_tokens=c["long_eval_tokens"]))
    bundle["reference_test_summary"] = {}
    for domain in evaluation_domains(data):
        questions = data[f"{domain}_test"]
        rr = evaluate(sampler, questions, 8, 420_000_000, OUT / f"reference_{domain}_test.json")
        stable = sum(sum(r["correct"] for r in rr[i*8:(i+1)*8]) >= 6 for i in range(len(questions)))
        bundle["reference_test_summary"][domain] = {"questions": len(questions), "initially_stable": stable,
            "accuracy": sum(r["correct"] for r in rr)/len(rr), "capped": sum(r["capped"] for r in rr)/len(rr)}
        print(f"[reference {domain}] stable={stable}/{len(questions)}; capped={bundle['reference_test_summary'][domain]['capped']:.2%}", flush=True)
        if stable < 128:
            print(f"[warning] {domain}: too few stable test questions for a precise forgetting estimate; more training steps do not fix this denominator.", flush=True)
        evaluate(sampler, questions, 4, 430_000_000, OUT / f"reference_{domain}_noise.json")
        evaluate(long_sampler, questions, 8, 450_000_000, OUT / f"reference_long_{domain}_test.json")
        evaluate(long_sampler, questions, 4, 440_000_000, OUT / f"reference_long_{domain}_noise.json")
    bundle["identity_sha256"] = digest(read(OUT / "identity.json"))
    write(OUT / "prepared.json", bundle)
    del sampler, long_sampler, model, tok; clear_memory()
    print(f"[prepare] B={bundle['budget']}; noise={bundle['noise_robust']:.3%}; epsilon={bundle['eps']:.3%}", flush=True)


def train(c, seed, arm):
    from engine import train_arm
    check_gpu(c)
    if not (SHARED / "prepared.json").exists():
        raise ValueError("Run prepare before training.")
    if c != read(SHARED / "config.json"):
        raise ValueError("Configuration changed after preparation; use a new experiment folder.")
    data, bundle = read(SHARED / "splits.json"), read(SHARED / "prepared.json")
    validate_splits(data, c)
    path = base_path(c)
    current = identity(c, checkpoint_identity(path), data)
    if digest(current) != bundle["identity_sha256"]:
        raise ValueError("Prepared reference/data/code/environment identity changed.")
    set_output(seed); OUT.mkdir(parents=True, exist_ok=True)
    c = dict(c, seed=seed, reference_checkpoint=path)
    freeze(OUT / "identity.json", identity(c, checkpoint_identity(path), data))
    train_arm(c, data, bundle, arm)


def main():
    from engine import ARMS
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="all", choices=["prepare", "train", "report", "all"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    c = read(ROOT / "config.json"); validate_config(c)
    if args.seed is not None and args.seed not in c["seeds"]:
        parser.error("Seed must be in config.json seeds")
    if args.command == "prepare":
        prepare(c)
    elif args.command == "train":
        if args.seed is None or args.arm is None:
            parser.error("train requires --seed and --arm")
        train(c, args.seed, args.arm)
    elif args.command == "report":
        from report import make_report, aggregate
        if args.seed is None:
            aggregate(c)
        else:
            set_output(args.seed); make_report(SHARED)
    else:
        # Separate processes release GPU memory between arms and keep resume paths explicit.
        subprocess.run([sys.executable, str(ROOT / "run.py"), "prepare"], check=True)
        for seed in c["seeds"]:
            for arm in ARMS:
                subprocess.run([sys.executable, str(ROOT / "run.py"), "train", "--seed", str(seed), "--arm", arm], check=True)
            subprocess.run([sys.executable, str(ROOT / "run.py"), "report", "--seed", str(seed)], check=True)
        from report import aggregate
        aggregate(c)


if __name__ == "__main__":
    main()
