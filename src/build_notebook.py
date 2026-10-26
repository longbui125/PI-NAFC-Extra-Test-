"""Build the standalone Colab notebook from the checked-in runtime."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import nbformat as nb

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT.parent/'notebooks/01_Phi_Olympiad_3Seeds.ipynb'


def sources():
    folder = ROOT/'phi_pilot'
    bodies = {p.name:p.read_bytes() for p in sorted(folder.iterdir())
              if p.suffix in ('.py','.json')}
    bodies['manifest.json'] = json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in bodies.items()},
                                        sort_keys=True,indent=2).encode()
    return bodies


def build(write=True):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
        for name,body in sorted(sources().items()):
            info = zipfile.ZipInfo(name,(2026,10,26,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info,body)
    payload = base64.b64encode(stream.getvalue()).decode()
    checksum = hashlib.sha256(stream.getvalue()).hexdigest()
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    cells = [md('''# Phi–Olympiad · 3 seeds

Chạy mới từ đầu: chuẩn bị dữ liệu → chọn anchor → hiệu chỉnh PI → train → đánh giá.
Chọn **A100**, **Run all**, cấp quyền Drive ở bước 2. Phi công khai; token có thể để trống.
Notebook nhúng mã và danh sách câu hỏi cố định, không cần repo hoặc ZIP của run cũ.

| Thành phần | Thiết lập |
|---|---|
| Model | microsoft/Phi-4-mini-instruct, 3.8B, revision cố định |
| Dữ liệu | NuminaMath-1.5: olympiads / olympiads_ref |
| Train / validation / test | 2.000 / 256 / 2.000 câu cố định |
| Seed | 1, 2, 3; chuẩn bị chung một lần, train độc lập |
| Nhánh mỗi seed | GRPO+PI → NAFC → quota replay → KL |
| Train | 300 bước/nhánh; 8 câu × 8 đáp án, micro-batch 1, LoRA rank 32 |
| PI | Cùng mục tiêu 400,86 token; eta 0,05; kp 4; trần lambda 3 |
| Context | Tổng 1.538 token như run seed 1; completion tối đa 1.536 token |
| Anchor | 128 câu đúng ≥6/8; 64 sửa, 64 đo; audit 4 mẫu mỗi 25 bước |
| Đánh giá | Validation tại 50/100/150/200/250; test ở 300; 4 mẫu/câu |

PI điều khiển **độ dài trung bình**, không cắt mọi câu ở 400,86 token.
Context gồm cả đề và đáp án nên đề dài làm giảm chỗ cho đáp án.
Quota dùng số nhóm replay từng bước của NAFC trong cùng seed, chọn anchor vòng tròn.
Đây là đối chứng chẩn đoán. KL beta = 0,04. Mọi nhánh bắt đầu từ model gốc.

**Lưu tự động:** checkpoint đầy đủ mỗi 25 bước, giữ 3 bản gần nhất và adapter các mốc đánh giá.
Có 12 lượt train; không bảo đảm hoàn thành trong một phiên Colab. Khi bị ngắt, mở lại notebook
và Run all trên cùng Drive: phần hoàn tất được kiểm tra và tái sử dụng.
Không chạy hai bản vào cùng folder. Notebook mới không nhập checkpoint của bản cũ.
'''), md('## 1. Token tùy chọn và mã thí nghiệm'), code('''import os
HF_TOKEN = "" # @param {type:"string"}
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
os.environ.update(USE_TF="0", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                  HF_HOME="/content/hf_cache", HF_HUB_DOWNLOAD_TIMEOUT="60", HF_HUB_ETAG_TIMEOUT="15")
'''), code('''import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
ROOT = Path("/content/nafc_phi_olympic")
ROOT.mkdir(parents=True,exist_ok=True)
PAYLOAD = "'''+payload+'''"
archive = base64.b64decode(PAYLOAD)
assert hashlib.sha256(archive).hexdigest() == "'''+checksum+'''"
with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
    for name in bundle.namelist():
        if Path(name).name != name:
            raise ValueError("Unsafe embedded path")
        (ROOT/name).write_bytes(bundle.read(name))
del archive, PAYLOAD
PLAN = json.loads((ROOT/"plan.json").read_text())
for name,expected in json.loads((ROOT/"manifest.json").read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import guard
guard.selfcheck(ROOT/"logs")
print("Embedded code and frozen questions verified.")
'''), md('''## 2. Cấp quyền Drive và cài thư viện

Đồng ý cấp quyền khi Colab hỏi. Kết quả lưu tại **My Drive/NAFC_Phi_Olympic_3Seeds_v2/**.
Chỉ cài thư viện khi thiếu hoặc sai phiên bản. Cài và kiểm tra trong tiến trình riêng để tránh
xung đột với kernel. Log hiện trực tiếp, có heartbeat và giới hạn chờ.
'''), code('''import subprocess
from google.colab import drive
gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True,timeout=15)
print(gpu)
if "A100" not in gpu:
    raise RuntimeError("Chọn A100 trước khi tiếp tục.")
drive.mount("/content/drive",timeout_ms=120000)
OUTPUT = Path("/content/drive/MyDrive/NAFC_Phi_Olympic_3Seeds_v2")
os.environ["PILOT_OUTPUT"] = str(OUTPUT)

def command(stage, *args, seed=None, timeout=21600, total_steps=None):
    options = list(map(str,args))
    folder = OUTPUT
    if seed is not None:
        options += ["--seed",str(seed)]
        folder = OUTPUT/f"seed_{seed}"
    logs = folder/"logs" if (folder/"identity.json").exists() else OUTPUT/"logs"
    name = "_".join([stage,*options]).replace("--","")
    guard.run([sys.executable,"-u",str(ROOT/"runner.py"),stage,*options],
              logs/(name+".log"),timeout=timeout,silence=900,cwd=ROOT,total_steps=total_steps)

probe = "import importlib.metadata as m; expected="+repr(PLAN["versions"])+"; assert all(m.version(k)==v for k,v in expected.items())"
installed = subprocess.run([sys.executable,"-c",probe],capture_output=True,timeout=30).returncode == 0
if not installed:
    guard.run([sys.executable,"-m","pip","install","--prefer-binary","--timeout","60","--retries","2",
               "--extra-index-url","https://download.pytorch.org/whl/cu128",
               *[f"{k}=={v}" for k,v in PLAN["versions"].items()]],
              OUTPUT/"logs/install.log",timeout=3600,silence=900)
guard.run([sys.executable,"-m","pip","uninstall","-y","torchao"],
          OUTPUT/"logs/remove_optional_torchao.log",timeout=120)
guard.run([sys.executable,"-u","-c",
    "import torch,vllm,trl,transformers,peft; from torch import nn; from peft import LoraConfig,get_peft_model; "
    "m=get_peft_model(nn.Sequential(nn.Linear(4,4)),LoraConfig(r=2,lora_alpha=2,target_modules=['0'])); "
    "m(torch.ones(1,4)).sum().backward(); assert any(p.grad is not None for p in m.parameters() if p.requires_grad); "
    "assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0); print('LoRA and A100 checks passed')"],
    OUTPUT/"logs/imports.log",timeout=300,silence=180)
'''), md('''## 3. Khởi tạo model và kiểm tra bộ nhớ

Tải checkpoint Phi cố định, xác minh câu hỏi và context. Chạy một lượt backward đủ độ dài
để kiểm tra bộ nhớ, không cập nhật trọng số. Lưu cả mã và cấu hình lên Drive.
'''), code('''command("initialize",timeout=3600)
command("preflight",timeout=1200)
'''), md('''## 4. Chuẩn bị dữ liệu và anchor chung

Model gốc sinh 8 mẫu/câu để đo độ đúng và hiệu chỉnh lambda PI trên train; mọi split dùng
ngân sách 400,86 token đã cố định từ calibration seed 1. Chọn 128 anchor đúng ≥6/8,
đo nhiễu bằng 4 mẫu độc lập. Train thử 2 bước trong `smoke/` riêng; model này không dùng cho các nhánh chính.
Model gốc được chấm lại độc lập 4 mẫu/câu để đo nhiễu đánh giá.

Mỗi lô 16 câu được lưu lên Drive. Lô đã lưu được dùng lại khi bị ngắt; lô đang dở phải chạy lại.
Ngưỡng kiểm tra năng lực và số anchor không tự hạ để ép thí nghiệm chạy.
'''), code('''command("readiness")
command("anchors")
command("prepare","--split","train")
command("smoke",timeout=1800,total_steps=2)
command("prepare","--split","validation")
command("prepare","--split","test")
command("base","--split","validation")
command("base","--split","test")
print("Shared preparation complete:",OUTPUT)
'''), md('''## 5. Train và đánh giá seed 1, 2, 3

Mỗi seed lần lượt chạy bốn nhánh. NAFC hoàn tất trước quota để đối chiếu lượng replay.
Mỗi nhánh 300 bước; đánh giá đúng các mốc quy định, không chọn checkpoint tốt nhất.
Checkpoint chứa optimizer, scheduler, RNG, PI, NAFC và vị trí replay. Chỉ dùng checkpoint
đã đánh dấu hoàn tất. Tối đa 24 bước chưa lưu có thể phải tính lại sau gián đoạn.

Train giới hạn 10 giờ/lần gọi; 15 phút không tiến thêm bước tối ưu sẽ dừng và giữ tiến độ đã lưu.
Heartbeat báo tiến trình còn được giám sát; cần nhìn số bước tăng để biết train có tiến triển.
'''), code('''for seed in PLAN["seeds"]:
    for arm in PLAN["arms"]:
        print(f"=== Seed {seed} · {arm} ===",flush=True)
        command("train","--arm",arm,seed=seed,timeout=36000,total_steps=300)
        for step in PLAN["keep_steps"]:
            split = "test" if step == PLAN["steps"] else "validation"
            command("merge","--arm",arm,"--step",step,seed=seed,timeout=1800)
            command("evaluate","--arm",arm,"--step",step,"--split",split,seed=seed)
    command("report",seed=seed,timeout=300)
    print(f"Seed {seed} complete; results and checkpoints are on Drive.",flush=True)
'''), md('## 6. Tổng hợp cả ba seed'), code('''command("aggregate",timeout=300)
print("Per-seed reports: seed_1/REPORT.md, seed_2/REPORT.md, seed_3/REPORT.md")
print("Three-seed mean and sample standard deviation: aggregate.csv and aggregate.json")
print("Drive:",OUTPUT)
'''), md('''## 7. Tải toàn bộ output

Drive đã lưu tự động. Nút dưới tạo một ZIP chứa dữ liệu chuẩn bị, đáp án sinh, log, báo cáo,
mã và checkpoint của cả ba seed. ZIP có thể lớn; chờ tải xong trước khi xóa runtime.
Không nén tự động khi Run all. Khi cần lưu giữa chừng, dừng cell train trước rồi chạy cell này.
'''), code('''import ipywidgets as widgets
from IPython.display import display
from google.colab import files
button = widgets.Button(description="Download output ZIP",layout=widgets.Layout(width="240px"))
status = widgets.Output()

def download_output(_):
    button.disabled = True
    try:
        with status:
            status.clear_output(wait=True)
            command("export",timeout=3600)
            files.download("/content/nafc_phi_olympic_3seeds.zip")
    finally:
        button.disabled = False

button.on_click(download_output)
display(button,status)
''')]
    book = nb.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'provenance':[]}})
    for index,cell in enumerate(book.cells):
        cell.id = f'phi-{index:02d}'
    nb.validate(book)
    if write:
        DESTINATION.parent.mkdir(parents=True,exist_ok=True)
        nb.write(book,DESTINATION)
        print(DESTINATION)
    return book


if __name__ == '__main__':
    build()
