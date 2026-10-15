"""Frozen Olympic train/holdout partitions, independent of training seed."""
import hashlib
import random
import re
from collections import Counter
from scoring import numeric_gold

DATASETS = {"numina": "AI-MO/NuminaMath-1.5", "omni": "KbsdJames/Omni-MATH"}


def normalized(text):
    return re.sub(r"[\s$\\{}]+", "", text.lower())


def rows(ds, domain, split, sources=None):
    result, seen = [], set()
    for i, item in enumerate(ds):
        if domain == "numina":
            if item["source"] not in sources or item["question_type"] != "math-word-problem":
                continue
            if any(str(item.get(k, "")).strip().lower() != "yes"
                   for k in ("problem_is_valid", "solution_is_valid")):
                continue
        answer = str(item.get("answer", "")).strip()
        if numeric_gold(answer) is None:
            continue
        question = item["problem"].strip()
        key = normalized(question)
        if not key or key in seen:
            continue
        seen.add(key)
        topics = item.get("domain", item.get("problem_type", "unknown"))
        topic = "; ".join(topics) if isinstance(topics, list) else str(topics)
        result.append({"id": hashlib.sha256(f"{domain}/{split}/{key}".encode()).hexdigest(),
                       "domain": domain, "question": question, "answer": answer,
                       "topic": topic, "source": item.get("source", domain),
                       "source_index": i, "difficulty": item.get("difficulty")})
    return result


def take(items, n, seed):
    if len(items) < n:
        raise ValueError(f"Need {n} eligible Olympic questions, available {len(items)}. Counts are not silently reduced.")
    return random.Random(seed).sample(items, n)


def evaluation_domains(data):
    return tuple(data["provenance"].get("evaluation_domains", ("numina", "omni")))


def validate_splits(data, c):
    counts = {"task_train": c["train_count"], "numina_validation": c["validation_count"],
              "numina_test": c["test_count"], "anchor_candidates": c["anchor_pool"]}
    seen_ids, seen_questions = set(), set()
    for name in (*counts, "omni_test"):
        if name in counts and len(data[name]) != counts[name]:
            raise ValueError(f"Wrong frozen count for {name}")
        if not data[name]:
            raise ValueError(f"Empty split: {name}")
        for q in data[name]:
            key = normalized(q["question"])
            if q["id"] in seen_ids or key in seen_questions:
                raise ValueError(f"Question leakage or duplicate in {name}")
            if not q["answer"]:
                raise ValueError("Empty gold answer")
            seen_ids.add(q["id"]); seen_questions.add(key)


def prepare_splits(c):
    from datasets import load_dataset
    print("[data] Loading Olympic-labelled Numina sources and Omni-MATH", flush=True)
    omni_raw = load_dataset(DATASETS["omni"], revision=c["omni_revision"], split="test")
    # Block every benchmark question, even ones outside our numeric-answer scoring scope.
    blocked = {normalized(q["problem"]) for q in omni_raw}
    omni = rows(omni_raw, "omni", "test")
    raw = load_dataset(DATASETS["numina"], revision=c["numina_revision"], split="train")
    allowed = rows(raw, "numina", "research_pool", c["numina_sources"])
    pool = [q for q in allowed if normalized(q["question"]) not in blocked]
    a = c["train_count"]; b = a + c["validation_count"]; d = b + c["test_count"]
    picked = take(pool, d + c["anchor_pool"], c["split_seed"])
    result = {"task_train": picked[:a], "numina_validation": picked[a:b],
              "numina_test": picked[b:d], "anchor_candidates": picked[d:], "omni_test": omni,
              "provenance": {"task_domain": "numina", "datasets": DATASETS,
                  "revisions": {k: c[f"{k}_revision"] for k in DATASETS},
                  "numina_sources": c["numina_sources"], "eligible_before_dedup": len(allowed),
                  "eligible_after_omni_dedup": len(pool), "omni_total": len(omni_raw),
                  "omni_scored": len(omni),
                  "split_status": "Numina research holdout; Omni numeric-answer subset, not the full official benchmark score.",
                  "filters": "Olympiad-labelled Numina sources; valid word problems; scalar numeric expressions only; no success/difficulty filter.",
                  "deduplication": "Normalized exact matching against ALL Omni questions. Paraphrase overlap is not certified."}}
    validate_splits(result, c)
    return result


def schedule(items, steps, prompts, seed):
    if len(items) < prompts:
        raise ValueError("Not enough unique questions for one batch")
    batches, cycle = [], 0
    while len(batches) < steps:
        epoch = list(items)
        random.Random(seed + cycle).shuffle(epoch)
        batches.extend(epoch[i:i + prompts] for i in range(0, len(epoch) - prompts + 1, prompts))
        cycle += 1
    return batches[:steps]


def describe(data):
    return {k: {"count": len(v), "topics": dict(Counter(q["topic"] for q in v)),
                "sources": dict(Counter(q["source"] for q in v))}
            for k, v in data.items() if k != "provenance"}
