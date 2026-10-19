"""Notebook setup: check the GPU before installing and reuse an already matching stack."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/content/drive/MyDrive/NAFC_Gemma_Olympiad_B756_seed1")
SOURCE_ROOT = Path("/content/drive/MyDrive/NAFC_Gemma_Olympiad_micro2")
ARCHIVES = Path("/content/nafc_archives")
VERSIONS = {"torch": "2.10.0+cu128", "vllm": "0.19.1", "trl": "1.13.0",
            "transformers": "5.18.0", "peft": "0.21.1"}


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def setup_step(label, command):
    log = ROOT/"logs/runtime_setup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[setup] {label}; log: {log}", flush=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {label}\n")
        out.flush()
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        next_update = started + 30
        try:
            while process.poll() is None:
                if time.monotonic() >= next_update:
                    print(f"[setup] {label}: {time.monotonic()-started:.0f}s elapsed", flush=True)
                    next_update = time.monotonic() + 30
                time.sleep(.25)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
    if process.returncode:
        with log.open(encoding="utf-8", errors="replace") as stream:
            from collections import deque
            print("".join(deque(stream, maxlen=60)), flush=True)
        raise RuntimeError(f"{label} failed ({process.returncode}); inspect {log}")
    print(f"[setup] {label}: done ({time.monotonic()-started:.1f}s)", flush=True)


def setup_runtime():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=15).strip()
    if "A100" not in gpu:
        raise RuntimeError(f"Select A100 before setup; received {gpu}. Nothing was installed.")
    print("[GPU]", gpu, flush=True)
    ROOT.mkdir(exist_ok=True)
    ARCHIVES.mkdir(exist_ok=True)
    identity_path = ROOT/"experiment_identity.json"
    if not identity_path.exists():
        identity_path = SOURCE_ROOT/"experiment_identity.json"
    if not identity_path.exists():
        raise FileNotFoundError(f"Mount the Drive containing the original prepared run: {SOURCE_ROOT}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected = {**VERSIONS, **{name: identity["software"][name] for name in ("math-verify", "datasets")}}
    missing = [name for name, version in expected.items() if installed(name) != version]
    if missing:
        print("[setup] Installing mismatched packages:", ", ".join(missing), flush=True)
        setup_step("Install pinned runtime", [sys.executable, "-m", "pip", "install", "--progress-bar", "off",
            "--extra-index-url", "https://download.pytorch.org/whl/cu128",
            *[f"{name}=={version}" for name, version in expected.items()]])
    else:
        print("[setup] All package versions already match; skipping pip install.", flush=True)
    if installed("torchao") is not None:
        setup_step("Remove unused torchao", [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    setup_step("CPU LoRA check", [sys.executable, "-c",
        "import torch; from torch import nn; from peft import LoraConfig, get_peft_model; "
        "model=get_peft_model(nn.Sequential(nn.Linear(4,4)), LoraConfig(r=2,lora_alpha=2,target_modules=['0'])); "
        "model(torch.ones(1,4)).sum().backward(); "
        "assert any(p.grad is not None for p in model.parameters() if p.requires_grad); "
        "print('CPU LoRA forward/backward passed')"])
    setup_step("GPU and library check", [sys.executable, "-c",
        "import torch,vllm,trl,transformers,peft; "
        "print(torch.__version__,vllm.__version__,trl.__version__,transformers.__version__,peft.__version__); "
        "assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'Select A100'"])
    if any(installed(name) != version for name, version in expected.items()):
        raise RuntimeError("Installed versions do not match the saved preparation; inspect runtime_setup.log")
    print("[setup] Ready. Continue to the embedded code cells; no manual runtime restart is requested.", flush=True)


if __name__ == "__main__":
    setup_runtime()
