"""Build only the Gemma notebook using the pinned, unchanged Ngoc scripts."""
import hashlib
import json
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parent


def build():
    source = ROOT/'gemma_ngoc'
    plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned source changed: '+name)
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    cells = [md('''# Gemma / Olympic mathematics: Ngoc-matched PI configuration

Configuration reference: DATN-FPTU main, commit `786291e`. Training, PI, NAFC and evaluation formulas follow Ngoc.
The model is Gemma 3-1B-IT. Data remains the previously selected Olympic experiment: NuminaMath training and
Omni-MATH evaluation. Ngoc's MATH questions are not used.

| Setting | Matched configuration |
|---|---|
| Arms | PI monitor, PI + NAFC, PI + fixed replay, PI + KL, PI + FlipGuard |
| Seeds | 1, 2, 3: 15 training runs, all reported |
| Train | The frozen 2,000 NuminaMath-1.5 numerical-answer questions from olympiads_ref / olympiads |
| Test | The frozen 2,000 numerical-answer Omni-MATH questions |
| Response cap | 1,536 tokens throughout; no longer-cap test |
| PI | Budget 358; eta 0.05; kp 4; signed lambda in [-3, 3] |
| Training | TRL 1.13 + vLLM 0.19.1; BF16; 300 steps; 8 prompts x 8 generations |
| LoRA / optimizer | r32 all linear layers; lr 1e-5; 10 warmup steps; micro-batch 4 |
| Anchors | Initial pool 800; extend to the pre-frozen 2,048 if insufficient; choose 128 at >=6/8 |
| Split / audit | 64 repair + 64 held, split seed 0; 4 draws every 25 steps |
| Controls | KL beta 0.04; FlipGuard gamma 0.1; fixed replay gamma 0.5, max fraction 0.5 |
| Tests | 4 draws at steps 50, 100, 150, 200, 300, evaluation seed 1 |
| Base reference | 8 draws, then independent 4-draw noise check, at 1,536 tokens only |

The frozen train/test partitions are preserved from the Olympic experiment, not replaced with Ngoc's MATH split.
The initial anchor pool is the first 800 candidates in the already frozen Numina candidate order. If it cannot
provide 128 eligible anchors, selection extends to the remaining candidates in the pre-frozen 2,048-question bank.
This resource adjustment follows the observed shortage of 66/128 in the initial pool; it is not an identical
800-candidate replication. Threshold, training data, test data and decoding remain fixed. Selection batches are
cached, eligible initial results are reused, and the final selected 128 anchors receive a fresh 4-draw noise check.
A data adapter supplies those rows; prepare_buckets adds only the Olympic dataset CLI choice.
Training passes a left-padded AutoTokenizer explicitly as TRL's processing_class, avoiding AutoProcessor's image
processor path for text-only Gemma. This changes input initialization, not the learning algorithm. Six upstream
scripts have no source-text edits. No Qwen scores, anchors, noise allowance or adapters
are reused. Omni-MATH is a competition-mathematics numeric subset, not an exclusively IMO benchmark or a full
official Omni-MATH score. Exact deduplication against the full Omni-MATH corpus was performed when freezing data.

The old output contains 64 validation questions, 128 anchors and 208 reference-test questions, all at 2,048 tokens.
It is preserved separately and is **not** resumed or pooled into this experiment. No optimizer training had started.
This new run must prepare Gemma-specific references and anchors with the upstream settings.

Select **A100**, fill the Hugging Face token below, then **Run all**. The notebook needs no repository or ZIP input
for a fresh run. Preparation includes upstream train-bucket sampling (2,000 x 8 responses) as well as test-reference
sampling (2,000 x [8 + 4]); these are shared by all 15 runs. No measured CU or single-session completion guarantee.
At the end, click **Download all output (.zip)**. Until downloaded, output exists only in the runtime.
'''), md('## 1. Access and optional resume'), code('''import os
HF_TOKEN = "" # @param {type:"string"}
RESTORE_MATCHED_OUTPUT = False # @param {type:"boolean"}
if not HF_TOKEN.strip():
    raise ValueError("Fill HF_TOKEN before Run all; accept the Gemma license on that Hugging Face account.")
os.environ["HF_TOKEN"] = HF_TOKEN.strip()
os.environ["USE_TF"] = "0"
os.environ["HF_HOME"] = "/content/hf_cache"
'''), md('''## 2. Install Ngoc's runtime

Training runs in fresh subprocesses, so they import the installed versions even if the notebook kernel previously
loaded other libraries. Math-Verify is installed without a version pin, as in Ngoc's notebook; its resolved version
is recorded and checked on resume. Do not bypass an installation or compatibility error.
The BF16 run does not use torchao quantization. Remove Colab's optional torchao package and verify a tiny CPU
LoRA forward/backward pass before loading Gemma, so incompatible preinstalled quantization hooks fail early.
'''), code('''import sys, subprocess
from pathlib import Path
ROOT = Path("/content/gemma_ngoc")
ROOT.mkdir(exist_ok=True)
ARCHIVES = Path("/content/nafc_archives")
ARCHIVES.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    "torch==2.10.0+cu128", "vllm==0.19.1", "trl==1.13.0", "transformers==5.18.0", "peft==0.21.1",
    "math-verify"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-c", "import torch; from torch import nn; from peft import LoraConfig, get_peft_model; model=get_peft_model(nn.Sequential(nn.Linear(4,4)), LoraConfig(r=2,lora_alpha=2,target_modules=['0'])); model(torch.ones(1,4)).sum().backward(); assert any(p.grad is not None for p in model.parameters() if p.requires_grad); print('CPU LoRA forward/backward passed')"], check=True)
subprocess.run([sys.executable, "-c", "import torch,vllm,trl,transformers,peft; print(torch.__version__,vllm.__version__,trl.__version__,transformers.__version__,peft.__version__); assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'Select A100'"], check=True)
'''), md('## 3. Embedded upstream code and matched plan')]
    names = [*plan['source_sha256'], 'plan.json']
    for name in names:
        body = (source/name).read_bytes().decode('utf-8')
        cells += [md(f'### `{name}`'), code(f'%%writefile /content/gemma_ngoc/{name}\n'+body)]
    cells += [md('''## 4. Verify sources and restore only matching output

For a new run, leave `RESTORE_MATCHED_OUTPUT=False`. For a later interrupted run, enable it and select the ZIP
downloaded from this matched notebook. Historical 2,048-token ZIPs are intentionally rejected.
'''), code('''import importlib.util, json, hashlib
from google.colab import files
# IPython adds a final newline to magic cells; restore bytes only when the pinned hash proves the match.
saved_plan = json.loads((ROOT/"plan.json").read_text(encoding="utf-8"))
for name, expected in saved_plan["source_sha256"].items():
    path = ROOT/name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        continue
    if data.endswith(b"\\n") and hashlib.sha256(data[:-1]).hexdigest() == expected:
        path.write_bytes(data[:-1])
    else:
        raise ValueError("Embedded source differs from its pinned hash: " + name)
spec = importlib.util.spec_from_file_location("gemma_matched_export", ROOT/"matched_run.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
PLAN = exporter.check_sources()
exporter.check_process_cleanup()
if RESTORE_MATCHED_OUTPUT:
    uploads = files.upload()
    if len(uploads) != 1:
        raise ValueError("Select one latest output ZIP from this Ngoc-matched Gemma notebook.")
    name, data = next(iter(uploads.items()))
    archive = ARCHIVES/"restore_gemma_matched.zip"
    archive.write_bytes(data)
    exporter.restore(archive)
    del uploads, data

def command(*args):
    print("Starting:", *args, flush=True)
    exporter.run_logged([sys.executable, "-u", str(ROOT/"matched_run.py"), *map(str, args)],
                        ROOT/"logs"/("command_"+"_".join(map(str,args)).replace("--", "")+".log"))

print("Matched plan: five arms x three seeds; 300 steps; 1,536-token cap; PI budget 358.")
'''), md('''## 5. Prepare once

Completed upstream preparation files are reused. An interrupted upstream generation command may need to restart
that command; there is no per-16-question HF resume in the unchanged vLLM scripts. Insufficient stable anchors stop
the experiment if the fixed maximum candidate bank is exhausted. The anchor count and success threshold are
never lowered. Selection results are saved after each 128-candidate batch; a partial batch may need to repeat.

The runner follows the parent process, not pipe EOF. Finished stages kill only their own process group. It prints
a heartbeat every minute and aborts a stage after 15 minutes without new output, preserving completed files and
checkpoints. This guard can also stop a legitimate long silent operation; inspect the log rather than bypassing it.
Output ZIPs from the previous matched Olympic notebook remain compatible; other experiment changes are rejected.
'''), code('command("prepare")'), md('''## 6. Run all prespecified arms and seeds

The scripts keep full checkpoints every 25 steps, retain the last three full checkpoints, and preserve milestone
adapters at steps 50-200. After each training run, evaluate its final model and every retained milestone.
All arms start from the unchanged Gemma checkpoint. No best-seed or best-step selection.
'''), code('''for seed in PLAN["seeds"]:
    for arm in ("mon", "rep", "fix", "kl", "fg"):
        print("Starting", arm, "seed", seed, flush=True)
        command("train", "--arm", arm, "--seed", seed)
'''), md('## 7. Report every arm, seed and milestone'), code('''command("report")
from IPython.display import display, Markdown
display(Markdown((ROOT/"REPORT.md").read_text(encoding="utf-8")))
'''), md('''## 8. Download all output

After an error or planned interruption, stop the running cell, wait for its subprocess to finish, then run this cell
manually. Download before disconnecting. ZIP includes preparation, results, response texts, logs, all remaining
full checkpoints and retained adapters. Model cache, token and merged model copies are excluded.
'''), code('''import ipywidgets as widgets
from IPython.display import display
button = widgets.Button(description="Download all output (.zip)", layout=widgets.Layout(width="260px"))
status = widgets.Output()

def download_output(_=None):
    button.disabled = True
    try:
        with status:
            status.clear_output(wait=True)
            print("Creating output archive; wait for download to finish...")
            archive = exporter.pack(ARCHIVES/"nafc_gemma_ngoc_matched_output.zip")
            files.download(str(archive))
    finally:
        button.disabled = False

button.on_click(download_output)
display(button, status)
''')]
    name = '01_Gemma_Olympiad_3Seeds.ipynb'
    book = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'name': name},
        'nafc': {'profile': 'gemma_ngoc_matched', 'upstream_commit': plan['upstream_commit'],
                 'seeds': plan['seeds'], 'historical_output_reused': False}})
    nb.validate(book)
    target = ROOT.parent/'notebooks'/name
    nb.write(book, target)
    return target


if __name__ == '__main__':
    print(build())
