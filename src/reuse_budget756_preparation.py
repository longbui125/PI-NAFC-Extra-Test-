"""Reuse validated Gemma predictions; change only train-budget metadata, never old runs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_ROOT = Path("/content/drive/MyDrive/NAFC_Gemma_Olympiad_micro2")
FILES = ("buckets_train_B440.json", "buckets_test_B440.json", "anchors.json", "results/base.json")


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def reuse_preparation(root, plan):
    root = Path(root).resolve()
    source = SOURCE_ROOT.resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("The B756 experiment must use a separate, non-nested output folder.")
    if plan["pi_budget"] != 756 or plan["seeds"] != [1]:
        raise ValueError("This preparation importer is only for B756, seed 1.")
    manifest_path = root/"preparation_reuse.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["source_plan_sha256"] != plan["preparation_source_plan_sha256"]:
            raise ValueError("The saved preparation has a different origin.")
        if not set(FILES).issubset(manifest["copied_sha256"]):
            raise ValueError("The saved preparation manifest is incomplete.")
        for name, version in manifest["source_software"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Use the preparation's {name}=={version}; rerun setup.")
        for name, digest in manifest["copied_sha256"].items():
            if not (root/name).is_file() or file_digest(root/name) != digest:
                raise ValueError("Reused preparation changed or is incomplete: "+name)
        if file_digest(root/"buckets_train_B756.json") != manifest["derived_sha256"]:
            raise ValueError("The B756 train metadata changed after preparation.")
        print("[reuse] Existing B756 preparation verified; no sampling or copying required.", flush=True)
        return
    if any((root/"runs").rglob("*")):
        raise ValueError("Training exists without preparation provenance; preserve it and inspect before proceeding.")
    identity_path = source/"experiment_identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"Original preparation not found: {source}. Mount the same Google Drive.")
    identity = read(identity_path)
    if plan_digest(identity["plan"]) != plan["preparation_source_plan_sha256"]:
        raise ValueError("The original Drive experiment does not match the pinned Gemma preparation plan.")
    for name, version in identity["software"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Use the preparation's {name}=={version}; rerun setup.")
    for name in FILES:
        if not (source/name).is_file():
            raise FileNotFoundError("Saved preparation is incomplete: "+str(source/name))
    frozen = read(root/"frozen_questions.json")
    for name, split in ((FILES[0], "train"), (FILES[1], "test")):
        data = read(source/name)
        expected = [(x["question"], x["answer"]) for x in frozen[split]]
        actual = [(x["question"], x["answer"]) for x in data["items"]]
        meta = data["meta"]
        if (actual != expected or len(actual) != 2000 or meta["max_len"] != 1536
                or meta["model"] != "/content/gemma_ngoc_model" or meta["dataset"] != "olympic"
                or meta["split"] != split or meta["n"] != 8 or meta["budget"] != 440):
            raise ValueError("Wrong model, split, cap, or frozen question order: "+name)
        for item in data["items"]:
            samples = item["samples"]
            if len(samples) != 8 or abs(sum(s[0] for s in samples)/8-item["p"]) > 1e-12:
                raise ValueError("Invalid cached correctness samples: "+name)
    anchors = read(source/"anchors.json")
    candidates = {(x["question"], x["answer"]) for x in frozen["anchor_pool"]}
    if len(anchors["items"]) != 128 or any((x["question"], x["answer"]) not in candidates for x in anchors["items"]):
        raise ValueError("The 128 anchors do not belong to the frozen Gemma candidate pool.")
    base = read(source/"results/base.json")
    if (len(base["per_question"]) != 2000 or base["model"] != "/content/gemma_ngoc_model"
            or base["decoding"] != {"temperature": 1.0, "top_p": 1.0, "n": 4, "seed": 1}):
        raise ValueError("The cached independent base evaluation does not match this experiment.")
    copied = {}
    names = list(FILES)
    if (source/"results/texts/base.jsonl.gz").is_file():
        names.append("results/texts/base.jsonl.gz")
    for name in names:
        original, target = source/name, root/name
        digest = file_digest(original)
        if target.exists() and file_digest(target) != digest:
            raise ValueError("Existing preparation differs; refusing to overwrite: "+name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+".tmp")
            shutil.copyfile(original, temporary)
            if file_digest(temporary) != digest:
                raise ValueError("Preparation changed during copying: "+name)
            temporary.replace(target)
        copied[name] = digest
    (root/"results/texts").mkdir(parents=True, exist_ok=True)
    target = root/"buckets_train_B756.json"
    temporary = target.with_suffix(target.suffix+".tmp")
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(root/"prepare_buckets.py"), "--reuse_samples",
        str(root/FILES[0]), "--budget", "756", "--out", str(temporary)], cwd=root, check=True)
    derived = read(temporary)
    if derived["items"] != read(root/FILES[0])["items"] or derived["meta"]["budget"] != 756:
        raise ValueError("Changing the budget must not change questions, samples or difficulty groups.")
    if target.exists() and target.read_bytes() != temporary.read_bytes():
        raise ValueError("Existing B756 metadata differs; refusing to overwrite.")
    temporary.replace(target)
    write_atomic(manifest_path, {"source_root": str(source),
        "source_plan_sha256": plan["preparation_source_plan_sha256"],
        "source_software": identity["software"], "copied_sha256": copied,
        "derived_sha256": file_digest(target), "training_checkpoints_reused": False})
    print("[reuse] Original samples, 128 anchors and base evaluation reused. Only train budget metadata is new.", flush=True)
