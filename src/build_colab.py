"""Build two self-contained Colab notebooks resuming the user's saved preparation."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat as nb

ROOT = Path(__file__).resolve().parent
COMMON = ('core.py', 'SOURCE.json', 'io_utils.py', 'scoring.py', 'data.py', 'model.py',
          'engine.py', 'report.py', 'colab_pilot.py', 'colab_report.py', 'colab_checks.py', 'manual_artifacts.py')
GEMMA_FIX = ('    if model.config.model_type == "gemma3_text":\n'
             '        model.generation_config.disable_compile = True  # Dynamo cannot compile the LoRA input-gradient hook.\n')


def load_progress(profile):
    if profile not in ('gemma', 'phi'): raise ValueError('Choose gemma or phi.')
    names = ('nafc_gemma_math_notebook01_v6.zip',) if profile == 'gemma' else (
        'nafc_phi_olympic_notebook02_v6.zip', 'nafc_phi_olympic_notebook03_v6.zip')
    paths = [ROOT.parent/'snapshots'/name for name in names]
    summaries = []
    for path in paths:
        with zipfile.ZipFile(path) as bundle:
            if len(bundle.namelist()) != len(set(bundle.namelist())):
                raise ValueError('Duplicate saved archive members.')
            if bundle.testzip() is not None: raise ValueError(f'Corrupt saved output: {path.name}')
            meta = json.loads(bundle.read('NAFC_BUNDLE.json'))
            if meta['identity'] != json.loads(bundle.read('shared/identity.json')):
                raise ValueError('Saved preparation identity is inconsistent.')
            summaries.append({name: hashlib.sha256(bundle.read(name)).hexdigest() for name in bundle.namelist()})
    if any(values != summaries[0] for values in summaries[1:]):
        raise ValueError('Phi saved outputs differ; refusing to discard or merge different preparation.')
    with zipfile.ZipFile(paths[0]) as bundle:
        meta = json.loads(bundle.read('NAFC_BUNDLE.json'))
        records = len(json.loads(bundle.read('shared/reference_validation.json'))['records'])
    config = json.loads((ROOT/('config_gemma_colab.json' if profile == 'gemma' else 'config_colab.json')).read_text(encoding='utf-8'))
    if meta['identity']['config'] != config or config['seeds'] != [42, 43, 44]:
        raise ValueError('Current configuration differs from the saved three-seed experiment.')
    return paths[0], meta, records, names


def frozen_sources(config, identity):
    bodies = {name: (ROOT/name).read_text(encoding='utf-8') for name in COMMON}
    bodies['config_colab.json'] = json.dumps(config, indent=2)+'\n'
    bodies[config['data_file']] = (ROOT/config['data_file']).read_text(encoding='utf-8')
    # Preserve Phi's saved source fingerprint; its original model loader predates the Gemma-only fix.
    expected = identity['code']['model.py']
    if hashlib.sha256(bodies['model.py'].encode()).hexdigest() != expected:
        candidate = bodies['model.py'].replace(GEMMA_FIX, '', 1)
        if config['test_domain'] != 'omni' or hashlib.sha256(candidate.encode()).hexdigest() != expected:
            raise ValueError('Cannot reconstruct the exact saved model loader.')
        bodies['model.py'] = candidate
    for name, digest in identity['code'].items():
        if hashlib.sha256(bodies[name].encode('utf-8')).hexdigest() != digest:
            raise ValueError(f'Source differs from saved run: {name}. Do not silently change resume identity.')
    return bodies


def build(profile='phi'):
    if profile == 'gemma':
        from build_gemma_ngoc import build as build_matched_gemma
        return build_matched_gemma()
    archive, meta, records, origins = load_progress(profile)
    config = meta['identity']['config']
    gemma = profile == 'gemma'
    slug = 'gemma_math' if gemma else 'phi_olympic'
    filename = '01_Gemma_MATH_3Seeds_Resume.ipynb' if gemma else '02_Phi_Olympiad_3Seeds_Resume.ipynb'
    title = 'Gemma 3-1B-IT / MATH' if gemma else 'Phi-4-mini-instruct / Numina to Omni-MATH'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    cells = [md(f'''# {title}: resume all three seeds

**Seeds: 42, 43, 44.** Two arms per seed: GRPO + PI (monitor), then GRPO + PI + NAFC.
This notebook embeds the saved preparation and every experiment source. No ZIP selection, repository access,
intermediate file transfer or other notebook is needed. Select **A100-SXM4-40GB**, then **Run all**.
{'Before Run all, paste your authorized Hugging Face read token into the token cell below. Accept the Gemma license on the same account first. Do not share a notebook containing your token.' if gemma else 'Phi is public; no token or login is required.'}

Saved validation progress: **{records//4}/64 questions** ({records} responses). Preparation and training are not complete yet.
{'The two old Phi ZIPs contain identical progress; one copy is embedded, without double-counting responses.' if not gemma else 'The saved Gemma loader already includes the generation-compilation fix.'}
The saved model/configuration/source fingerprints are preserved. Software and GPU identity must still match.

Both new notebooks can run independently at the same time. Phi prepares once for all three training seeds,
saving the duplicated preparation of the previous two-Phi-notebook layout. It still runs six arms sequentially;
fewer concurrent GPUs do not halve the total training work. Completion within 500 CU or one session is unmeasured.

| Setting | Preserved protocol |
|---|---|
| Train / validation / test questions | 2,000 / 64 / 2,000 |
| Steps / responses | 300 steps per arm; 8 prompts x 8 fresh responses per step |
| Anchors | Exactly 128; 64 repair and 64 held controller |
| Selection / calibration | >=6/8 initial successes; three independent four-response noise audits |
| Online audits | Four responses on each anchor every 25 steps |
| Primary tests | Steps 50, 100, 150, 200, 300; four responses/question |
| Primary / secondary cap | {config['max_new_tokens']} / {config['long_eval_tokens']} tokens; secondary at step 300 only |
| Reference | Eight responses plus independent four-response noise check at both caps |
| Training | LoRA r32, all linear layers, lr 1e-5, fp16 |

PI/NAFC formulas, data and evaluation budgets are unchanged. All milestones and capped/unboxed outputs stay in the report.
{'MATH includes all seven topics and answer types; source rows with missing gold are logged.' if gemma else 'Omni-MATH uses the frozen numeric-answer subset, not the complete official benchmark or exclusively IMO problems.'}

Output remains local. At the end, click **Download all output (.zip)**. The archive includes results and full resume
checkpoints, but no model cache or credentials. A destroyed runtime loses progress that has not been downloaded.
''')]
    if gemma:
        cells += [md('''## 1. Hugging Face access - fill before Run all

Accept [Gemma's license](https://huggingface.co/google/gemma-3-1b-it) and create a
[read token](https://huggingface.co/settings/tokens) on that account. Paste it between the quotes below.
No Secrets or interactive input prompt is used. The token remains in the notebook if you save it;
remove it before sharing. It is not included in the output ZIP.
'''), code('''import os
HF_TOKEN = "" # @param {type:"string"}
if not HF_TOKEN.strip():
    raise ValueError("Fill HF_TOKEN in this cell before Run all. Use a read token authorized for Gemma.")
os.environ["HF_TOKEN"] = HF_TOKEN.strip()
''')]
    versions = meta['identity']['versions']
    pinned = [f'{name}=={version}' for name, version in versions.items() if name != 'torch']
    cells += [md('''## 2. Runtime setup

The saved CUDA PyTorch build must match; other recorded libraries are installed at their saved versions.
Do not replace the embedded sources or change the configuration of this resumed run.
'''), code(f'''import os, sys, subprocess, time
import importlib.metadata
from pathlib import Path
ROOT = Path("/content/nafc_a100")
ROOT.mkdir(parents=True, exist_ok=True)
os.environ["USE_TF"] = "0"
os.environ["HF_HOME"] = "/content/hf_cache"
EXPECTED_VERSIONS = {versions!r}
if importlib.metadata.version("torch") != EXPECTED_VERSIONS["torch"]:
    raise RuntimeError("This saved run needs PyTorch " + EXPECTED_VERSIONS["torch"] + ". Select the matching Colab runtime; do not bypass the saved experiment identity.")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *{pinned!r}], check=True)
ASSIGNED_SEEDS = [42, 43, 44]
SLUG = {slug!r}
OUTPUT = Path("/content/results") / (SLUG + "_resume_3seeds")
OUTPUT.mkdir(parents=True, exist_ok=True)
ARCHIVES = Path("/content/nafc_archives")
ARCHIVES.mkdir(parents=True, exist_ok=True)
'''), md('## 3. Experiment sources and saved preparation')]
    for name, body in frozen_sources(config, meta['identity']).items():
        cells += [md(f'### `{name}`'), code(f'%%writefile /content/nafc_a100/{name}\n'+body)]
    payload = base64.b64encode(archive.read_bytes()).decode('ascii')
    payload_cell = code('# @title Embedded saved preparation (do not edit)\nEMBEDDED_PROGRESS = (\n'+
        '\n'.join(repr(payload[i:i+4096]) for i in range(0,len(payload),4096))+'\n)')
    payload_cell.metadata['cellView'] = 'form'
    cells += [md(f'''### Saved output embedded in this notebook

Source: `{archive.name}`; SHA-256: `{digest}`. This is ZIP data, not hidden experiment code.
The cell below restores the saved validation responses automatically. It never adds duplicate Phi responses.
'''), payload_cell, md('## 4. Automatic restore and checks'), code(f'''import base64, hashlib, json, zipfile
from google.colab import files
sys.path.insert(0, str(ROOT))
from manual_artifacts import pack, unpack
CONFIG = json.loads((ROOT/"config_colab.json").read_text(encoding="utf-8"))
archive = ARCHIVES/"{archive.name}"
payload = base64.b64decode(EMBEDDED_PROGRESS, validate=True)
if hashlib.sha256(payload).hexdigest() != {digest!r}:
    raise ValueError("Embedded preparation ZIP is corrupt; do not start a different run.")
archive.write_bytes(payload)
del payload
with zipfile.ZipFile(archive) as bundle:
    saved_identity = json.loads(bundle.read("NAFC_BUNDLE.json"))["identity"]
if CONFIG != saved_identity["config"]:
    raise ValueError("Configuration changed from the saved run.")
for name, expected in saved_identity["code"].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
        raise ValueError("Experiment source changed: " + name)
for package, expected in saved_identity["versions"].items():
    if importlib.metadata.version(package) != expected:
        raise ValueError("Software version mismatch: " + package + "; expected " + expected)
identity_path = OUTPUT/"shared"/"identity.json"
if identity_path.exists():
    if json.loads(identity_path.read_text(encoding="utf-8")) != saved_identity:
        raise ValueError("Existing local output belongs to a different experiment.")
    print("Using current local progress; embedded older progress will not overwrite it.")
else:
    print("Restored saved files:", unpack([archive], OUTPUT, CONFIG))
state = json.loads((OUTPUT/"shared"/"reference_validation.json").read_text(encoding="utf-8"))
print("Saved validation questions:", len(state["records"])//4, "/", CONFIG["validation_count"])
if CONFIG["base_model"].startswith("google/"):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    try:
        hf_hub_download(CONFIG["base_model"], "config.json", revision=CONFIG["base_revision"], token=os.environ["HF_TOKEN"])
    except HfHubHTTPError:
        raise RuntimeError("Gemma access failed. Accept its license and fill a read token from the same account in section 1.") from None
subprocess.run([sys.executable, str(ROOT/"colab_checks.py")], cwd=ROOT, check=True)


def command(*args):
    started = time.monotonic()
    cmd = [sys.executable, "-u", str(ROOT/"colab_pilot.py"), *args, "--output", str(OUTPUT)]
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.kill(); process.wait()
            raise
    if code: raise RuntimeError(f"{{args[0]}} failed (exit {{code}}); see the original traceback above.")
    print(f"[{{args[0]}}] elapsed {{(time.monotonic()-started)/3600:.2f}} session hours.")
'''), md('''## 5. Continue preparation

Completed validation chunks are reused. Model loading and the backward preflight repeat; saved responses do not.
After validation, preparation selects anchors, calibrates noise and evaluates the frozen reference.
This is still preparation, not optimizer training.
'''), code('''print("Continue saved preparation; preparation seed:", CONFIG["seed"])
command("prepare")
'''), md('''## 6. Train all three seeds

Each seed starts independently from the unchanged base model. Completed arms are skipped on compatible reruns.
If anchor selection is inconclusive, no training is performed and the diagnostic remains available for download.
'''), code('''if (OUTPUT/"shared"/"prepared.json").exists():
    for seed in ASSIGNED_SEEDS:
        for arm in ("grpo_pi", "grpo_pi_nafc"):
            command("train", "--seed", str(seed), "--arm", arm)
        print("Paired seed complete:", seed)
else:
    print("Preparation is inconclusive; no arm was trained. Download the diagnostic below.")
'''), md('## 7. Report all three seeds'), code('''from IPython.display import display, Markdown, Image
ARMS = ("grpo_pi", "grpo_pi_nafc")
complete = all((OUTPUT/f"seed_{seed}"/arm/"complete.json").exists()
               for seed in ASSIGNED_SEEDS for arm in ARMS)
if complete:
    command("report")
    display(Markdown((OUTPUT/"REPORT.md").read_text(encoding="utf-8")))
    display(Image(filename=str(OUTPUT/"overview.png")))
    display(Image(filename=str(OUTPUT/"trajectory.png")))
else:
    print("Aggregate report pending; all three paired seeds are required. Partial output is downloadable below.")
'''), md('''## 8. Download all output

Click the button when the run finishes. It creates one ZIP with all available results and full resume state.
After an interruption, wait for the process to stop and execute this cell manually to download partial output.
Wait for ZIP creation and browser download to finish before ending the runtime.
'''), code('''import ipywidgets as widgets
from IPython.display import display

if not (OUTPUT/"shared"/"identity.json").exists():
    print("No restored experiment output yet.")
else:
    download_button = widgets.Button(description="Download all output (.zip)", layout=widgets.Layout(width="260px"))
    download_status = widgets.Output()

    def download_output(_=None):
        download_button.disabled = True
        try:
            with download_status:
                download_status.clear_output(wait=True)
                print("Creating full output ZIP...")
                archive = pack(OUTPUT, ARCHIVES/f"nafc_{SLUG}_3seeds_output.zip", seed="all")
                print("ZIP ready:", archive, f"({archive.stat().st_size/1024**3:.2f} GiB)")
                files.download(str(archive))
        finally:
            download_button.disabled = False

    download_button.on_click(download_output)
    display(download_button, download_status)
''')]
    book = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}, 'accelerator':'GPU', 'colab':{'name':filename},
        'nafc':{'profile':profile,'assigned_seeds':config['seeds'],'embedded_progress_sha256':digest,
                'saved_validation_questions':records//4,'source_archives':list(origins)}})
    nb.validate(book)
    target = ROOT.parent/'notebooks'/filename
    target.parent.mkdir(exist_ok=True)
    nb.write(book,target); print(target)
    return target


if __name__ == '__main__':
    build('gemma'); build('phi')
