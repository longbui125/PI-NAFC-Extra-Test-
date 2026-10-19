"""A separate five-arm, seed-1 budget ablation; original notebooks and runs stay intact."""
import hashlib
import json
from pathlib import Path

import nbformat as nb

from build_gemma_ngoc import build as build_original

ROOT = Path(__file__).resolve().parent
DRIVE_ROOT = "/content/drive/MyDrive/NAFC_Gemma_Olympiad_B756_seed1"
OLD_ROOT = "/content/drive/MyDrive/NAFC_Gemma_Olympiad_micro2"


def build():
    book = build_original("gemma", write=False)
    bodies = {c.source.split("\n", 1)[0].rsplit("/", 1)[-1]: c.source.split("\n", 1)[1]
              for c in book.cells if c.cell_type == "code" and c.source.startswith("%%writefile")}
    plan = json.loads(bodies["plan.json"])
    source_digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan.update(seeds=[1], pi_budget=756,
        preparation_source_plan_sha256=source_digest,
        experiment="Exploratory PI-budget ablation: seed 1, five arms, B756",
        preparation_reused="Frozen B440 train/test samples, anchors and base evaluation from the B358 Gemma run; no training state.")
    runner = bodies["matched_run.py"].replace("buckets_train_B358.json", "buckets_train_B756.json")
    start, end = runner.index("def prepare():"), runner.index("\ndef evaluate_model(")
    runner = runner[:start]+'''def prepare():
    p = check_sources()
    from reuse_budget756_preparation import reuse_preparation
    reuse_preparation(ROOT, p)
    initialize()
    print('Preparation verified: no GPU sampling; B756; five arms; seed 1 only.', flush=True)

'''+runner[end:]
    runner = runner.replace("# Gemma / Olympic mathematics: Ngoc-matched PI configuration",
                            "# Gemma / Olympic mathematics: seed 1, PI budget 756")
    runner = runner.replace("All five arms and all three seeds are reported.",
                            "All five arms of seed 1 are reported. This budget was chosen after viewing the original run.")
    runner = runner.replace("'comparison.json', 'REPORT.md'", "'comparison.json', 'REPORT.md', 'preparation_reuse.json'")
    bodies["matched_run.py"] = runner
    bodies["reuse_budget756_preparation.py"] = (ROOT/"reuse_budget756_preparation.py").read_text(encoding="utf-8")
    for name in ("matched_run.py", "reuse_budget756_preparation.py"):
        plan["source_sha256"][name] = hashlib.sha256(bodies[name].encode()).hexdigest()
    bodies["plan.json"] = json.dumps(plan, ensure_ascii=False, indent=2)+"\n"
    for cell in book.cells:
        if cell.cell_type == "code":
            if cell.source.startswith("%%writefile"):
                name = cell.source.split("\n", 1)[0].rsplit("/", 1)[-1]
                cell.source = "%%writefile "+DRIVE_ROOT+"/"+name+"\n"+bodies[name]
            else:
                cell.source = cell.source.replace(OLD_ROOT, DRIVE_ROOT).replace(
                    "five arms x three seeds; 300 steps; 1,536-token cap; PI budget 358.",
                    "five arms x seed 1; 300 steps; 1,536-token cap; PI budget 756.").replace(
                    "nafc_gemma_ngoc_matched_output.zip", "nafc_gemma_olympiad_B756_seed1_output.zip")
    plan_index = next(i for i,c in enumerate(book.cells) if c.cell_type == "code"
                      and c.source.startswith("%%writefile "+DRIVE_ROOT+"/plan.json\n"))
    book.cells[plan_index:plan_index] = [nb.v4.new_markdown_cell("### Preparation reuse helper"),
        nb.v4.new_code_cell("%%writefile "+DRIVE_ROOT+"/reuse_budget756_preparation.py\n"+bodies["reuse_budget756_preparation.py"])]
    book.cells[0].source = '''# Gemma / Olympic mathematics — seed 1, PI budget 756

Five arms: GRPO+PI monitor, NAFC, fixed replay, KL (beta 0.04), FlipGuard (gamma 0.1).
Each starts from the unchanged public Gemma 3-1B-IT model and runs 300 optimizer steps with training seed 1.
This is an exploratory follow-up to the B358 experiment, not an additional seed of that experiment.

Only the training PI budget changes: 358 -> 756. Model revision, frozen 2,000 training and 2,000 test questions,
NAFC, 128 anchors (64 repair / 64 held), noise allowance, LoRA, optimizer, batch size and decoding are unchanged.
The completion cap stays 1,536 tokens. Checkpoints are tested at 50, 100, 150, 200 and the prespecified final 300.
Test difficulty groups and token-budget metrics retain the original B440 test reference, shared across experiments.

756 is about 65% of Gemma's original mean train completion length (1,162 tokens), matching the approximate
358/550 ratio in Ngoc's Qwen-MATH run. It is a hypothesis to test, not a guarantee of better results.

Select A100, enter your Hugging Face token below, then Run all and authorize the SAME Drive as the old run.
Preparation is copied from My Drive/NAFC_Gemma_Olympiad_micro2, never generated again. No ZIP upload is needed.
If required preparation is missing or incompatible, the notebook stops instead of resampling it.
Old trained adapters, optimizers and PI/NAFC states are not reused. Old results/checkpoints are never overwritten.

New output and resumable checkpoints are saved under My Drive/NAFC_Gemma_Olympiad_B756_seed1.
After a VM replacement, mount the same Drive and Run all; completed new arms are skipped and partial new arms
resume their latest complete checkpoint. Do not run two copies against the same new output folder.
Runtime installation is skipped when versions already match. A genuinely fresh VM still needs library/model
downloads; Colab session continuity or completion within a CU budget is not guaranteed.
'''
    install_index = next(i for i,c in enumerate(book.cells) if c.cell_type == "code" and '"pip", "install"' in c.source)
    book.cells[install_index-1].source = '''## 2. Check A100 and reuse the installed runtime

Wrong GPU stops before any installation. Matching packages skip pip entirely. Otherwise only the mismatched
packages and their required dependencies need installation. math-verify/datasets versions match the saved
preparation. A tiny CPU LoRA check and a fresh-process GPU/import check run before model loading.
Progress appears every 30 seconds; the complete log persists in the NEW experiment's logs/runtime_setup.log.
Returning successfully does not request a runtime restart. Reconnection or VM replacement is controlled by Colab.
'''
    book.cells[install_index].source = (ROOT/"budget756_runtime.py").read_text(encoding="utf-8")
    for cell in book.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## 5."):
            cell.source = '''## 5. Reuse saved preparation — no reference/anchor generation

Validate the original plan, software versions and frozen question order. Copy existing train/test predictions,
128 anchors and the independent base evaluation into the new folder. The original CPU-only --reuse_samples
path derives B756 training metadata without calling a model. Its samples and difficulty groups must stay exact.
A provenance manifest records file hashes. On restart it verifies the saved copy and does not need to copy again.
Only then are the pinned public base model files downloaded/reused from the local Hugging Face cache.
'''
        elif cell.cell_type == "markdown" and cell.source.startswith("## 4."):
            cell.source = '''## 4. Verify code and optionally restore this NEW experiment

Leave RESTORE_MATCHED_OUTPUT=False when the new output is already on Drive, or on the first run.
Old B358 preparation is read automatically in section 5; do not upload an old B358 ZIP here.
Only ZIPs exported by this B756 seed-1 notebook are accepted for optional restore.
'''
    name = "03_Gemma_Olympiad_B756_Seed1.ipynb"
    book.metadata.colab.name = name
    book.metadata.nafc.update(profile="gemma_budget756_seed1", seeds=[1], historical_output_reused=True)
    for cell in book.cells:
        cell.id = hashlib.sha256((cell.cell_type+"\0"+cell.source).encode()).hexdigest()[:12]
    nb.validate(book)
    target = ROOT.parent/"notebooks"/name
    nb.write(book, target)
    return target


if __name__ == "__main__":
    print(build())
