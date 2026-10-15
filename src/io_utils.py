"""Small file helpers and immutable run identity."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(f.name, path)
    finally:
        if Path(f.name).exists():
            Path(f.name).unlink()


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Different frozen experiment: {path}. Use a fresh outputs directory.")
    else:
        write(path, value)


def checkpoint_identity(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("Reference checkpoint is missing; run shared preparation first.")
    config_name = "config.json"
    weights = sorted(folder.glob("*.safetensors"))
    if not weights or not (folder / config_name).exists():
        raise ValueError("Reference needs config.json and safetensors weights.")
    c = read(folder / config_name)
    if c.get("model_type") not in ("phi3", "gemma3_text"):
        raise ValueError("This experiment requires a pinned Phi or Gemma checkpoint.")
    files = weights + [folder / config_name]
    files += [p for p in folder.iterdir() if p.is_file() and
              ("tokenizer" in p.name or p.name in ("vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json")
               or p.name.endswith("safetensors.index.json"))]
    return {"kind": "merged", "files_sha256": {p.name: file_hash(p) for p in files}}


def validate_config(c):
    if c["base_model"] != "microsoft/Phi-4-mini-instruct":
        raise ValueError("Do not change the backbone inside a frozen pilot.")
    for k in ("steps", "prompts_per_step", "group_size", "train_count", "validation_count", "test_count",
              "anchor_count", "anchor_pool", "anchor_selection_batch", "initial_draws", "eval_draws", "generation_batch", "audit_every",
              "warmup_steps", "max_new_tokens", "max_prompt_tokens", "lora_rank", "long_eval_tokens"):
        if not isinstance(c[k], int) or c[k] <= 0:
            raise ValueError(f"{k} must be a positive integer")
    if c["group_size"] < 2 or c["initial_draws"] != 8 or c["eval_draws"] != 4:
        raise ValueError("The frozen protocol uses group size >=2, 8 initial draws and 4 evaluation draws.")
    if c["anchor_count"] % 2 or c["anchor_pool"] < c["anchor_count"]:
        raise ValueError("Anchors must split equally into repair/control halves.")
    if c["train_count"] % c["prompts_per_step"]:
        raise ValueError("train_count must be divisible by prompts_per_step to preserve complete prompt groups.")
    if not 0 < c["compression_ratio"] < 1 or c["precision"] not in ("fp16", "bf16"):
        raise ValueError("Invalid compression ratio or precision")
    if any(s <= 0 or s >= c["steps"] for s in c["validation_steps"]):
        raise ValueError("Validation milestones must precede the final step.")
    if not 0 <= c["noise_margin"] < 1 or not 0 < c["replay_frac_max"] < 1:
        raise ValueError("Invalid noise margin or replay fraction")
    if not 0 < c["fixed_replay_fraction"] <= c["replay_frac_max"]:
        raise ValueError("Fixed replay must use the same maximum replacement budget.")
    if c["long_eval_tokens"] <= c["max_new_tokens"]:
        raise ValueError("Long evaluation must exceed the primary token cap.")
    if len(set(c["seeds"])) != len(c["seeds"]) or len(c["seeds"]) < 2 or c["seed"] not in c["seeds"]:
        raise ValueError("Use the prespecified independent training seeds.")
    for k in ("learning_rate", "pi_eta", "pi_kp", "pi_lambda_max", "nafc_eta", "nafc_lambda_max"):
        if not isinstance(c[k], (int, float)) or not 0 < c[k] < float("inf"):
            raise ValueError(f"Invalid {k}")


def experiment_identity(c, parent, splits, environment):
    settings = {k: v for k, v in c.items() if k != "reference_checkpoint"}
    code = {p.name: file_hash(p) for p in sorted(ROOT.glob("*.py"))}
    return {"protocol": "phi-olympic-pi-nafc-v1", "settings": settings,
            "reference": parent, "splits_sha256": digest(splits), "code_sha256": code,
            "source": read(ROOT / "SOURCE.json"), "environment": environment}
