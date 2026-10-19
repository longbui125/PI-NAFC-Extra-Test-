"""Budget ablation must reuse predictions, isolate outputs and avoid repeated setup installs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import nbformat
import pytest

from build_budget756 import build, DRIVE_ROOT, OLD_ROOT

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/"src/gemma_ngoc"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def new_run(tmp_path, monkeypatch):
    book = nbformat.read(build(), as_version=4)
    root = tmp_path/"B756"; root.mkdir()
    for cell in book.cells:
        if cell.cell_type == "code" and cell.source.startswith("%%writefile"):
            header, body = cell.source.split("\n", 1)
            (root/header.rsplit("/",1)[-1]).write_bytes(body.encode("utf-8"))
    monkeypatch.syspath_prepend(str(root))
    # Each case must load its own helper, not a previous fixture's cached module.
    monkeypatch.delitem(sys.modules,"reuse_budget756_preparation",raising=False)
    runner = load_module("budget756_runner", root/"matched_run.py")
    helper = load_module("budget756_helper", root/"reuse_budget756_preparation.py")
    return book, runner, helper


def test_new_notebook_changes_only_budget_seed_and_preparation_routing(new_run):
    book, runner, helper = new_run
    plan=runner.check_sources(); old=json.loads((SOURCE/"plan.json").read_text(encoding="utf-8"))
    assert plan["pi_budget"]==756 and plan["seeds"]==[1] and plan["steps"]==300
    changed={"pi_budget","seeds","source_sha256","preparation_source_plan_sha256","experiment","preparation_reused"}
    assert {k:v for k,v in plan.items() if k not in changed}=={k:v for k,v in old.items() if k not in changed}
    assert plan["preparation_source_plan_sha256"]==helper.plan_digest(old)
    assert runner.MODEL==Path("/content/gemma_ngoc_model")
    for name in old["source_sha256"]:
        if name!="matched_run.py":
            assert (runner.ROOT/name).read_bytes()==(SOURCE/name).read_bytes()
    for cell in book.cells:
        if cell.cell_type!="code": continue
        assert cell.outputs==[] and cell.execution_count is None
        if cell.source.startswith("%%writefile"):
            assert cell.source.split("\n",1)[0].startswith("%%writefile "+DRIVE_ROOT+"/")
            if cell.source.split("\n",1)[0].endswith(".py"):
                compile(cell.source.split("\n",1)[1],"embedded","exec")
        else:
            compile(cell.source,"notebook_cell","exec")
    loop=next(c.source for c in book.cells if c.cell_type=="code" and 'for seed in PLAN[' in c.source)
    calls=[]; exec(loop,dict(PLAN=plan,command=lambda *a:calls.append(a)))
    assert calls==[("train","--arm",arm,"--seed",1) for arm in ("mon","rep","fix","kl","fg")]
    for arm in runner.ARMS:
        args=runner.train_args(arm,1)
        assert args[args.index("--buckets")+1]=="buckets_train_B756.json"
        assert Path(args[args.index("--model")+1])==Path("/content/gemma_ngoc_model")
        assert "--resume" not in args and args[args.index("--max_completion_length")+1]=="1536"
        assert args[args.index("--micro_batch")+1]=="2"
    with pytest.raises(ValueError,match="Undeclared"):
        runner.train_args("rep",2)


@pytest.fixture
def saved_preparation(new_run,tmp_path,monkeypatch):
    book, runner, helper = new_run
    archive=PROJECT/"snapshots/nafc_gemma_ngoc_matched_output.zip"
    if not archive.exists(): pytest.skip("User's preparation ZIP is intentionally not published")
    source=tmp_path/"original";source.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for name in helper.FILES:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(bundle.read(name))
    old=json.loads((SOURCE/"plan.json").read_text(encoding="utf-8"))
    software={**old["versions"],"math-verify":"0.9.0","datasets":"4.8.5"}
    (source/"experiment_identity.json").write_text(json.dumps({"plan":old,"software":software}),encoding="utf-8")
    monkeypatch.setattr(helper,"SOURCE_ROOT",source)
    original_version=helper.importlib.metadata.version
    monkeypatch.setattr(helper.importlib.metadata,"version",lambda name:software[name] if name in software else original_version(name))
    monkeypatch.setenv("PYTHONIOENCODING","utf-8")
    (source/"runs/old_seed1/checkpoint-300").mkdir(parents=True)
    (source/"runs/old_seed1/checkpoint-300/adapter_model.safetensors").write_bytes(b"old trained weights must not be reused")
    return book,runner,helper,source


def test_actual_preparation_reused_without_sampling_or_old_training_state(saved_preparation,monkeypatch):
    _,runner,helper,source=saved_preparation
    original={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    helper.reuse_preparation(runner.ROOT,runner.check_sources())
    for name in helper.FILES:
        assert (runner.ROOT/name).read_bytes()==(source/name).read_bytes()
    new=helper.read(runner.ROOT/"buckets_train_B756.json")
    assert new["items"]==helper.read(source/"buckets_train_B440.json")["items"]
    assert new["budget"]==756 and new["meta"]["budget"]==756
    assert not (runner.ROOT/"runs").exists()
    assert original=={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(helper.subprocess,"run",lambda *a,**kw:pytest.fail("Completed preparation regenerated"))
    monkeypatch.setattr(helper,"SOURCE_ROOT",source.parent/"not_mounted_now")
    helper.reuse_preparation(runner.ROOT,runner.check_sources())


@pytest.mark.parametrize("problem",["wrong_model","question_order","missing_anchors","old_training_in_new_folder"])
def test_bad_preparation_stops_instead_of_sampling_or_overwriting(saved_preparation,problem):
    _,runner,helper,source=saved_preparation
    path=source/"buckets_train_B440.json"
    if problem=="wrong_model":
        identity=helper.read(source/"experiment_identity.json");identity["plan"]["model"]="a different model"
        (source/"experiment_identity.json").write_text(json.dumps(identity),encoding="utf-8")
    elif problem=="question_order":
        data=helper.read(path);data["items"][0],data["items"][1]=data["items"][1],data["items"][0]
        path.write_text(json.dumps(data),encoding="utf-8")
    elif problem=="missing_anchors":
        (source/"anchors.json").unlink()
    else:
        folder=runner.ROOT/"runs/old";folder.mkdir(parents=True);(folder/"optimizer.pt").write_bytes(b"old")
    with pytest.raises((ValueError,FileNotFoundError)):
        helper.reuse_preparation(runner.ROOT,runner.check_sources())
    assert not (runner.ROOT/"preparation_reuse.json").exists()


def test_prepare_reuses_before_model_download_and_never_calls_gpu_sampler(new_run,monkeypatch):
    _,runner,_=new_run
    import types
    calls=[]
    monkeypatch.setitem(sys.modules,"reuse_budget756_preparation",types.SimpleNamespace(reuse_preparation=lambda *a:calls.append("reuse")))
    monkeypatch.setattr(runner,"initialize",lambda:calls.append("model_download"))
    monkeypatch.setattr(runner,"execute",lambda *a:pytest.fail("GPU preparation unexpectedly requested"))
    runner.prepare()
    assert calls==["reuse","model_download"]


def runtime_module(tmp_path,monkeypatch):
    m=load_module("budget_runtime",PROJECT/"src/budget756_runtime.py")
    monkeypatch.setattr(m,"ROOT",tmp_path/"new")
    monkeypatch.setattr(m,"SOURCE_ROOT",tmp_path/"old")
    monkeypatch.setattr(m,"ARCHIVES",tmp_path/"archives")
    m.SOURCE_ROOT.mkdir()
    versions={**m.VERSIONS,"math-verify":"0.9.0","datasets":"4.8.5"}
    (m.SOURCE_ROOT/"experiment_identity.json").write_text(json.dumps({"software":versions}),encoding="utf-8")
    monkeypatch.setattr(m.subprocess,"check_output",lambda *a,**kw:"NVIDIA A100-SXM4-40GB\n")
    return m,versions


def test_matching_runtime_skips_install_and_unused_uninstall(tmp_path,monkeypatch):
    m,versions=runtime_module(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(m,"installed",versions.get)
    monkeypatch.setattr(m,"setup_step",lambda label,command:calls.append((label,command)))
    m.setup_runtime()
    assert [label for label,_ in calls]==["CPU LoRA check","GPU and library check"]
    assert all("pip" not in command for _,command in calls)


def test_wrong_gpu_stops_before_install_or_new_output(tmp_path,monkeypatch):
    m,_=runtime_module(tmp_path,monkeypatch)
    monkeypatch.setattr(m.subprocess,"check_output",lambda *a,**kw:"NVIDIA L4\n")
    monkeypatch.setattr(m,"setup_step",lambda *a:pytest.fail("Wrong GPU started setup"))
    with pytest.raises(RuntimeError,match="Nothing was installed"):
        m.setup_runtime()
    assert not m.ROOT.exists()


def test_fresh_runtime_pins_saved_resolved_dependencies_and_removes_torchao(tmp_path,monkeypatch):
    m,versions=runtime_module(tmp_path,monkeypatch);installed={"torchao":"0.10.0"};calls=[]
    monkeypatch.setattr(m,"installed",installed.get)
    def run(label,command):
        calls.append((label,command))
        if "install" in command: installed.update(versions)
        if "uninstall" in command: installed.pop("torchao")
    monkeypatch.setattr(m,"setup_step",run)
    m.setup_runtime()
    assert "math-verify==0.9.0" in calls[0][1] and "datasets==4.8.5" in calls[0][1]
    assert calls[1][1][-3:]==["uninstall","-y","torchao"]


def test_setup_logs_subprocess_failure_and_does_not_wait_for_pipe_eof(tmp_path,monkeypatch):
    m,_=runtime_module(tmp_path,monkeypatch)
    m.setup_step("success",[sys.executable,"-c","print('persisted setup output')"])
    with pytest.raises(RuntimeError,match=r"failed \(7\)"):
        m.setup_step("failure",[sys.executable,"-c","print('failure evidence'); raise SystemExit(7)"])
    text=(m.ROOT/"logs/runtime_setup.log").read_text(encoding="utf-8")
    assert "persisted setup output" in text and "failure evidence" in text


def test_old_budget_zip_cannot_restore_training_into_new_experiment(new_run,tmp_path):
    _,runner,_=new_run
    old=json.loads((SOURCE/"plan.json").read_text(encoding="utf-8"))
    path=tmp_path/"old.zip"
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("MATCHED_RUN.json",json.dumps({"plan":old}))
        z.writestr("runs/lacPIrepH_s1/optimizer.pt",b"old")
    with pytest.raises(ValueError,match="Different experiment"):
        runner.restore(path)
    assert not (runner.ROOT/"runs").exists()
