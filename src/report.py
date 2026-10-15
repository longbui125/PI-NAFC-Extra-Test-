"""Paired per-seed results and uncertainty across all prespecified seeds."""
import json
import numpy as np
from io_utils import ROOT, OUT, read, write
from data import evaluation_domains


def question_rates(records, questions, n):
    if len(records) != len(questions)*n:
        raise ValueError("Incomplete evaluation; do not compare partial arms.")
    fields = {k: [] for k in ("accuracy", "tokens", "capped", "no_box")}
    for i, q in enumerate(questions):
        group = records[i*n:(i+1)*n]
        if [(r["id"], r["sample_index"]) for r in group] != [(q["id"], j) for j in range(n)]:
            raise ValueError("Evaluation question/sample order mismatch")
        fields["accuracy"].append(sum(r["correct"] for r in group)/n)
        fields["tokens"].append(sum(len(r["tokens"]) for r in group)/n)
        for k in ("capped", "no_box"):
            fields[k].append(sum(r[k] for r in group)/n)
    return {k: np.asarray(v) for k, v in fields.items()}


def retention(initial, current, cohort=None):
    cohort = initial >= .75 if cohort is None else cohort
    n = int(cohort.sum()); bad = int((cohort & (current <= .25)).sum())
    return {"initially_stable": n, "correct_to_low_success": bad, "rate": bad/n if n else None}


def paired_interval(values, seed=0):
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    # Chunk bootstrap draws to avoid allocating a samples-by-full-benchmark matrix.
    means = [values[rng.integers(0, len(values), (100, len(values)))].mean(axis=1) for _ in range(40)]
    return np.percentile(np.concatenate(means), [2.5, 97.5]).tolist()


def make_report(shared):
    from engine import ARMS
    data = read(shared / "splits.json")
    bundle = read(shared / "prepared.json")
    payload = {"seed": read(OUT / "identity.json")["settings"]["seed"],
               "data_provenance": data["provenance"], "results": {}, "differences": {}, "warnings": []}
    for arm in ARMS:
        if not (OUT / arm / "complete.json").exists():
            raise ValueError(f"{arm} incomplete; all three arms are required.")
        payload["results"][arm] = {"run": read(OUT / arm / "complete.json")}
    for limit in ("primary", "long"):
        prefix = "" if limit == "primary" else "long_"
        for domain in evaluation_domains(data):
            questions = data[f"{domain}_test"]
            initial = question_rates(read(shared / f"reference_{prefix}{domain}_test.json")["records"], questions, 8)
            noise = question_rates(read(shared / f"reference_{prefix}{domain}_noise.json")["records"], questions, 4)
            stable = initial["accuracy"] >= .75
            raw = {}
            endpoint = f"{limit}_{domain}"
            payload[endpoint] = {"reference": {k: float(v.mean()) for k, v in initial.items()},
                                 "unchanged_noise": retention(initial["accuracy"], noise["accuracy"])}
            for arm in ARMS:
                name = "final" if limit == "primary" else "long"
                current = question_rates(read(OUT / arm / f"{name}_{domain}_test.json")["records"], questions, 4)
                raw[arm] = current
                gained_cohort = initial["accuracy"] <= .25
                gained_n = int(gained_cohort.sum())
                payload["results"][arm][endpoint] = {
                    **{k: float(v.mean()) for k, v in current.items()},
                    "retention": retention(initial["accuracy"], current["accuracy"]),
                    "newly_correct": int((gained_cohort & (current["accuracy"] >= .75)).sum()),
                    "initially_low_success": gained_n,
                    "regressions_by_topic": {topic: retention(initial["accuracy"], current["accuracy"],
                        stable & np.asarray([q["topic"] == topic for q in questions]))
                        for topic in sorted({q["topic"] for q in questions})}}
            for control in ARMS[:-1]:
                a, b = raw["grpo_pi_nafc"]["accuracy"], raw[control]["accuracy"]
                flip = (a[stable] <= .25).astype(float) - (b[stable] <= .25).astype(float)
                payload["differences"][f"{endpoint}_vs_{control}"] = {
                    "accuracy": float((a-b).mean()), "accuracy_question_ci95": paired_interval(a-b),
                    "flip_rate": float(flip.mean()) if len(flip) else None,
                    "flip_question_ci95": paired_interval(flip)}
            if stable.sum() < 128:
                payload["warnings"].append(f"{endpoint}: only {stable.sum()} initially stable test questions; coarse retention estimate.")
    if payload["results"]["grpo_pi_nafc"]["run"]["anchor_prompts"] == 0:
        payload["warnings"].append("NAFC never replayed; this seed cannot demonstrate active protection.")
    for arm in ARMS:
        info = payload["results"][arm]["run"]
        if info["optimizer_updates"] != info["step"]:
            payload["warnings"].append(f"{arm}: skipped optimizer updates; examine logs.")
    payload["warnings"] += ["Omni scores cover the numeric-answer subset; not the full official leaderboard protocol.",
        "Numina is a research holdout. Olympic source labels are not a certificate that every question is IMO-level.",
        "Deduplication does not certify absence of paraphrases or model pretraining contamination.",
        "Fixed replay uses one prespecified fraction, not a best-of-grid baseline."]
    write(OUT / "comparison.json", payload)
    lines = ["# Phi Olympic results", "", f"Training seed: {payload['seed']}. Final checkpoint only.", "",
             "| Endpoint | Arm | Accuracy | Correct → low success | Tokens | Capped |",
             "|---|---|---:|---:|---:|---:|"]
    for endpoint in ("primary_numina", "primary_omni", "long_numina", "long_omni"):
        for arm in ARMS:
            r = payload["results"][arm][endpoint]; f = r["retention"]
            lines.append(f"| {endpoint} | {arm} | {r['accuracy']:.2%} | {f['correct_to_low_success']}/{f['initially_stable']} | {r['tokens']:.1f} | {r['capped']:.2%} |")
    lines += ["", "Capped and unboxed responses remain in every primary denominator.",
              f"Noise allowance epsilon={bundle['eps']:.3%}; PI budget B={bundle['budget']}.",
              "Question intervals are conditional on this training seed; use SUMMARY.md for seed variation.",
              "", *[f"- {w}" for w in payload["warnings"]]]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plots(payload)
    print("\n".join(lines), flush=True)
    return payload


def plots(payload):
    from engine import ARMS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, domain in zip(axes, ("numina", "omni")):
        endpoint = f"primary_{domain}"
        ax.bar(["Reference", *ARMS], [100*payload[endpoint]["reference"]["accuracy"],
               *[100*payload["results"][a][endpoint]["accuracy"] for a in ARMS]])
        ax.set(title=f"{domain}: numeric-answer subset", ylabel="Accuracy (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(OUT / "accuracy.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, domain in zip(axes, ("numina", "omni")):
        for arm in ARMS:
            r = payload["results"][arm][f"primary_{domain}"]["retention"]
            ax.bar(arm, r["correct_to_low_success"])
        ax.set(title=f"{domain}: initially stable n={r['initially_stable']}", ylabel="Regressed questions")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(OUT / "regressions.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm in ARMS:
        m = read(OUT / arm / "train_metrics.json"); a = read(OUT / arm / "audit_metrics.json")
        axes[0].plot([r["step"] for r in m], [r["pi_lambda_used"] for r in m], label=arm)
        axes[1].plot([r["step"] for r in a], [r["F_used"] for r in a], label=arm)
        axes[2].plot([r["step"] for r in a], [r["lam"] for r in a], label=arm)
    for ax, title in zip(axes, ("PI multiplier", "Held-anchor flagged rate", "NAFC multiplier")):
        ax.set(title=title, xlabel="Training batch"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "controllers.png", dpi=160); plt.close(fig)


def aggregate(c):
    root = ROOT / "outputs_phi_olympic"
    summaries = []
    for seed in c["seeds"]:
        path = root / f"seed_{seed}" / "comparison.json"
        if not path.exists():
            raise ValueError(f"Seed {seed} is incomplete. All prespecified seeds are required.")
        summaries.append(read(path))
    fingerprints = [read(root / f"seed_{s}" / "identity.json") for s in c["seeds"]]
    settings = [{k: v for k, v in x["settings"].items() if k not in ("seed", "reference_checkpoint")} for x in fingerprints]
    if any(x != settings[0] for x in settings) or any(
            (x["splits_sha256"], x["reference"], x["code_sha256"]) !=
            (fingerprints[0]["splits_sha256"], fingerprints[0]["reference"], fingerprints[0]["code_sha256"])
            for x in fingerprints):
        raise ValueError("Cannot aggregate different protocols, splits, references or code.")
    results = {}
    for endpoint in summaries[0]["differences"]:
        results[endpoint] = {}
        for metric in ("accuracy", "flip_rate"):
            values = [s["differences"][endpoint][metric] for s in summaries]
            if any(v is None for v in values):
                results[endpoint][metric] = {"mean": None, "ci95": None, "seeds": values}
            else:
                v = np.asarray(values)
                results[endpoint][metric] = {"mean": float(v.mean()), "ci95": paired_interval(v), "seeds": values}
    write(root / "summary.json", {"seeds": c["seeds"], "differences_nafc_minus_baseline": results,
        "scope": "Seed-bootstrap intervals conditional on one frozen data split. Five seeds do not establish universal effectiveness."})
    lines = ["# Phi Olympic multi-seed summary", "", f"All prespecified seeds: {c['seeds']}.", "",
             "| Comparison | Accuracy delta (pp) | 95% seed CI | Regression-rate delta (pp) | 95% seed CI |",
             "|---|---:|---|---:|---|"]
    def estimate(r):
        return "undefined" if r["mean"] is None else f"{100*r['mean']:+.2f}"
    def interval(r):
        return "undefined" if r["ci95"] is None else f"[{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]"
    for key, r in results.items():
        lines.append(f"| {key} | {estimate(r['accuracy'])} | {interval(r['accuracy'])} | {estimate(r['flip_rate'])} | {interval(r['flip_rate'])} |")
    lines += ["", "Negative regression delta favours NAFC. Positive accuracy delta favours NAFC.",
              "Inspect unchanged-reference noise, active replay, task exposure, token cost and long-cap sensitivity before interpreting protection.",
              "No seed removal or best-checkpoint selection is used. These intervals are descriptive with only five independent seeds."]
    (root / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
