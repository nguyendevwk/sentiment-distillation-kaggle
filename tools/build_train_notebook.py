"""Build the standalone Kaggle distillation pilot (no download/install/model run)."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(s).strip() + '\n'})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(s).strip() + '\n',
                  'execution_count': None, 'outputs': []})


md('''
# Qwen3 distillation — corpus teacher cố định, Kaggle pilot

Upload notebook này lên Kaggle, bật **GPU + Internet**. Không cần clone repo hoặc GitHub token.
Source train đã được đóng gói. Notebook tải duy nhất file teacher Qwen v2; không tải/chạy teacher.

Mặc định: Qwen3-0.6B, 5.000 mẫu train tối đa sau lọc, 200 bước mỗi nhánh, LoRA rank 16,
sequence tối đa 2.048, micro-batch 1 × accumulation 8. Chạy **uniform** và **progress** tuần tự.
Đây là pilot để kiểm tra phương pháp, không phải tái lập paper hay kết quả benchmark cuối.
Đổi sang Qwen3-1.7B ở cell cấu hình để thử model thứ hai.

Pipeline: tải → audit → xác nhận mapping → prepare/split → kiểm tra runtime → train → so sánh log → export.
**Không Run All ngay lần đầu**: đọc dữ liệu tại bước audit và xác nhận prompt/response trước khi prepare.
Không có kết quả train Kaggle được xác nhận trước; thời gian/VRAM phụ thuộc GPU và token thực tế.
''')
code('''
from pathlib import Path
import json, os, sys, subprocess, hashlib, shutil
BASE = Path('/kaggle/working/distill-pilot' if Path('/kaggle').exists()
            else '/content/distill-pilot' if Path('/content').exists() else './distill-pilot').resolve()
BASE.mkdir(parents=True, exist_ok=True)
MODEL = 'Qwen/Qwen3-0.6B'  # Lần sau: 'Qwen/Qwen3-1.7B'
MODEL_REVISION = 'main'  # prepare sẽ resolve SHA cụ thể và lưu vào manifest
DATASET = 'zhang-yice/sentiment-distillation-v2'
DATASET_REVISION = '0ee48fc022bb45ad9f2c66c43fc1c73db6c3f4fb'
TEACHER_FILE = 'Qwen3-32B_50k.jsonl'
LOCAL_TEACHER_FILE = None  # Hoặc '/kaggle/input/.../Qwen3-32B_50k.jsonl'
INSTALL_DEPENDENCIES = True  # Giữ CUDA torch của Kaggle; có dry-run bảo vệ torch
FOUR_BIT = False  # LoRA thường trước; True thêm bitsandbytes và bật QLoRA
RUN_TAG = 'pilot-v1'  # Đổi tên nếu thay cấu hình; không ghi đè thí nghiệm cũ
MAX_TRAIN = 5000  # 0 = toàn bộ train hợp lệ sau grouped holdout
MAX_LENGTH = 2048  # Mẫu dài hơn bị loại nguyên mẫu, không cắt đáp án
STEPS = 200
MODES = ['uniform', 'progress']  # Ablation: thêm 'balanced', 'loss'
SEED = 42
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
PROBE_EVERY = 25
SAVE_EVERY = 100
RESUME = {}  # Ví dụ {'progress': '/kaggle/working/.../progress/checkpoint-100'}
EXPORT_MODE = None  # Đặt 'uniform' hoặc 'progress' sau khi train xong
MODEL_TAG = MODEL.split('/')[-1]
WORK = BASE / RUN_TAG / MODEL_TAG
WORK.mkdir(parents=True, exist_ok=True)
print('Work:', WORK)
''')
bundle_files = [
    'training/distill_core.py', 'training/train_distill.py', 'requirements-train-kaggle.txt',
    'kaggle_evaluate.py', 'icl_multifaced_kaggle.py', 'fsa_kaggle.py',
    'notebook_runtime.py', 'model_name_kaggle.json'
]
bundle = {p: (ROOT / p).read_text(encoding='utf-8') for p in bundle_files}
code('BUNDLED = ' + repr(bundle) + '''
for relative, source in BUNDLED.items():
    target = BASE / 'source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
SCRIPT = BASE / 'source/training/train_distill.py'
REQUIREMENTS = BASE / 'source/requirements-train-kaggle.txt'

def run(arguments, log=None):
    command = [sys.executable, str(SCRIPT), *map(str, arguments)]
    if log is None:
        subprocess.run(command, check=True)
        return
    # Stream output and persist it; failed child raises instead of continuing to next mode.
    with Path(log).open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
''')
md('''
## Dependency trên Kaggle

Cell cài thư viện train vào runtime hiện tại; không nâng/hạ PyTorch/CUDA.
Pip dry-run dừng nếu kế hoạch thay `torch`, `torchvision`, `torchaudio`, `triton`, hoặc `nvidia-*`.
Mọi code model chạy ở subprocess mới, tránh module cache của notebook sau pip install.
Không cần Hugging Face token cho Qwen public; nếu bị rate-limit, dùng secret `HF_TOKEN`.
''')
code('''
if INSTALL_DEPENDENCIES:
    # Gỡ bỏ torchao lỗi thời của Kaggle để tránh xung đột với peft
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
    extra = ['bitsandbytes==0.46.1'] if FOUR_BIT else []
    report = WORK / 'pip-plan.json'
    base = [sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '-r', str(REQUIREMENTS), *extra]
    subprocess.run([*base, '--dry-run', '--report', str(report)], check=True)
    plan = json.loads(report.read_text())
    names = [p['metadata']['name'].lower().replace('_', '-') for p in plan.get('install', [])]
    forbidden = [n for n in names if n in {'torch', 'torchvision', 'torchaudio', 'triton'} or n.startswith('nvidia-')]
    if forbidden:
        raise RuntimeError(f'Pip would replace CUDA stack: {forbidden}; use a compatible platform runtime.')
    subprocess.run(base, check=True)
else:
    print('Using platform packages; the next runtime cell checks imports.')
''')
code('''
# GPU/dependency validation only; no synthetic model inference.
subprocess.run([sys.executable, '-c', r"""
import json, torch, transformers, peft, accelerate
from importlib.metadata import version
from packaging.version import Version
assert Version(transformers.__version__) >= Version('4.51.0'), 'Qwen3 requires transformers >= 4.51'
assert torch.cuda.is_available(), 'Enable a GPU accelerator'
print(json.dumps({
    'versions': {p: version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'huggingface-hub')},
    'gpus': [{'name': torch.cuda.get_device_name(i),
              'vram_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
             for i in range(torch.cuda.device_count())],
    'bf16': torch.cuda.is_bf16_supported(),
}, indent=2))
"""], check=True)
print('Pilot dùng GPU 0; không tự gộp VRAM của hai T4.')
''')
md('''
## Tải đúng file và audit

Dùng `hf_hub_download` cho từng file, tránh lỗi schema khi loader trộn ba teacher.
Revision được khóa; file attach thủ công vẫn được kiểm SHA-256 ở prepare nhưng cần tự xác minh provenance.
Audit lấy mẫu reservoir trải khắp file; không chỉ xem những dòng đầu thường cùng task.
''')
code('''
from huggingface_hub import HfApi, hf_hub_download
if LOCAL_TEACHER_FILE:
    CORPUS = Path(LOCAL_TEACHER_FILE)
    CORPUS_REVISION = 'local-unverified-provenance:' + DATASET_REVISION
else:
    CORPUS_REVISION = HfApi().dataset_info(DATASET, revision=DATASET_REVISION).sha
    CORPUS = Path(hf_hub_download(DATASET, TEACHER_FILE, repo_type='dataset', revision=CORPUS_REVISION))
run(['audit', '--file', CORPUS, '--group_field', 'selected_cluster_id', '--examples', 40,
     '--output', WORK / 'audit.json'], log=WORK / 'audit.log')
''')
md('''
## Xác nhận field mapping sau khi đọc audit

Đọc tối thiểu 30–50 mẫu: user text đã nằm trong `instruction` chưa? `input` rỗng có nghĩa gì?
`PROMPT_FIELDS` là thứ tự nối field bằng hai newline; không được nhân đôi user text hoặc demonstration.
Ví dụ **chỉ khi đã xác minh**: `['instruction', 'text']` nếu instruction không chứa câu hỏi cuối;
hoặc `['instruction', 'input']` nếu input là phần cần nối; hoặc `['instruction']` nếu đã chứa đầy đủ.
Giữ nguyên field gốc và đáp án. Không parse/viết lại response teacher.

`SOURCE_FIELD` phải là văn bản nguồn dùng phát hiện cùng input qua nhiều task; không dùng task_name/ID.
`GROUP_FIELD` cần đủ mẫu/nhóm; `cluster_id` có thể rỗng, kiểm tra `selected_cluster_id` trong audit.
Đặt `MAPPING_CONFIRMED=True` sau khi kiểm tra. Đây là bước chặn train sai schema, không tự suy đoán.
''')
code('''
PROMPT_FIELDS = ['instruction', 'text']
RESPONSE_FIELD = 'Qwen/Qwen3-32B_output'
GROUP_FIELD = 'selected_cluster_id'
SOURCE_FIELD = 'text'
MAPPING_CONFIRMED = True

if not MAPPING_CONFIRMED or not PROMPT_FIELDS:
    raise ValueError('Read audit.json, set PROMPT_FIELDS and confirm mapping before prepare.')
PREPARED = WORK / 'prepared'
prepare_args = ['prepare', '--file', CORPUS, '--output', PREPARED, '--model', MODEL,
                '--model_revision', MODEL_REVISION, '--corpus_revision', CORPUS_REVISION,
                '--prompt_fields', ','.join(PROMPT_FIELDS), '--response_field', RESPONSE_FIELD,
                '--group_field', GROUP_FIELD, '--source_field', SOURCE_FIELD, '--mapping_confirmed',
                '--max_length', MAX_LENGTH, '--max_train', MAX_TRAIN, '--seed', SEED,
                '--max_groups', 16, '--probe_per_group', 4, '--holdout_per_group', 4]
request = dict(model=MODEL, model_revision=MODEL_REVISION, corpus_revision=CORPUS_REVISION,
               file=str(CORPUS), prompt_fields=PROMPT_FIELDS, response_field=RESPONSE_FIELD,
               group_field=GROUP_FIELD, source_field=SOURCE_FIELD, max_length=MAX_LENGTH,
               max_train=MAX_TRAIN, seed=SEED)
request_file = WORK / 'prepare-request.json'
if (PREPARED / 'manifest.json').exists():
    if not request_file.exists() or json.loads(request_file.read_text()) != request:
        raise ValueError('Prepare settings differ. Change RUN_TAG instead of reusing old splits.')
    print('Reusing prepared corpus; train checks split hashes.')
else:
    run(prepare_args, log=WORK / 'prepare.log')
    request_file.write_text(json.dumps(request, indent=2))
manifest = json.loads((PREPARED / 'manifest.json').read_text())
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print((PREPARED / 'preview.json').read_text())
''')
md('''
## Train baseline và đề xuất

Uniform và progress dùng cùng samples, shuffle, seed, LoRA, steps, response-only mean-per-example loss.
Mọi nhánh đều đo probe/holdout cùng lịch để kiểm soát chi phí đo. Progress cập nhật trọng số bằng EMA
của tiến bộ tương đối trên probe, hỗn hợp với trọng số đều; giới hạn 0,5–2 và chuẩn hóa theo train counts.
Probe được dùng điều phối; holdout tách input độc lập và chỉ ghi log, không cập nhật controller.
Không chọn cấu hình bằng SentiBench-test. Nhóm không có probe vẫn giữ trọng số nền; xem coverage.

Checkpoint gồm adapter, optimizer, scheduler, scaler, RNG, data cursor và controller.
Để resume: restore toàn thư mục WORK vào cùng đường dẫn, giữ source/config/packages và điền RESUME.
Không chỉ restore adapter. Hai nhánh mặc định 200 bước vẫn chỉ là pilot, chưa đi qua toàn train pool.
''')
code('''
for mode in MODES:
    output = WORK / mode
    run_request = dict(model=MODEL, prepared=str(PREPARED), mode=mode, steps=STEPS,
                       batch_size=1, grad_accum=GRAD_ACCUM, learning_rate=LEARNING_RATE,
                       rank=16, seed=SEED, probe_every=PROBE_EVERY,
                       warmup_steps=PROBE_EVERY, save_every=SAVE_EVERY, four_bit=FOUR_BIT)
    request_path = WORK / f'{mode}-request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != run_request:
        raise ValueError('Train settings changed. Choose a new RUN_TAG; do not silently skip an old run.')
    request_path.write_text(json.dumps(run_request, indent=2))
    if (output / 'summary.json').exists():
        print('Completed already:', mode, '— choose a new RUN_TAG to rerun with changed settings.')
        continue
    arguments = ['train', '--prepared', PREPARED, '--output', output, '--mode', mode,
                 '--steps', STEPS, '--batch_size', 1, '--grad_accum', GRAD_ACCUM,
                 '--learning_rate', LEARNING_RATE, '--rank', 16, '--seed', SEED,
                 '--probe_every', PROBE_EVERY, '--warmup_steps', PROBE_EVERY,
                 '--save_every', SAVE_EVERY]
    if FOUR_BIT:
        arguments.append('--four_bit')
    if RESUME.get(mode):
        arguments += ['--resume', RESUME[mode]]
    run(arguments, log=WORK / f'{mode}.log')
''')
md('''
## Số liệu pilot

CE là mức khớp đáp án teacher, **không phải Accuracy/Macro-F1/SentiBench score**.
Thời gian gồm train, probe, holdout và các checkpoint trước đó; VRAM ghi peak tensor allocated,
không thay cho tổng VRAM hiển thị bởi nvidia-smi. So token và optimizer_updates giữa các nhánh;
FP16 có thể bỏ optimizer update khi overflow — nếu khác nhiều cần chỉnh LR/precision và chạy lại.
Để tuyên bố cải thiện chất lượng, phải đánh giá checkpoint trên benchmark với evaluator đã kiểm chứng.
''')
code('''
rows = []
for mode in MODES:
    path = WORK / mode / 'summary.json'
    if not path.exists():
        continue
    summary = json.loads(path.read_text())
    state = summary['state']
    events = [json.loads(s) for s in (WORK / mode / 'metrics.jsonl').read_text().splitlines() if s.strip()]
    measured = [r for r in events if r.get('holdout_loss')]
    counts = manifest['holdout_groups']
    def holdout_mean(event):
        values = event['holdout_loss']
        return sum(values[g] * counts[g] for g in values) / sum(counts[g] for g in values)
    rows.append({'mode': mode, 'steps': state['step'], 'updates': state['optimizer_updates'],
                 'initial_holdout_ce': holdout_mean(measured[0]),
                 'final_holdout_ce': holdout_mean(measured[-1]),
                 'train_tokens': state['train_tokens'], 'probe_tokens': state['probe_tokens'],
                 'holdout_tokens': state['holdout_tokens'],
                 'minutes': state['elapsed_seconds'] / 60,
                 'peak_allocated_gib': state['peak_vram_bytes'] / 2**30})
print(json.dumps(rows, indent=2))
(WORK / 'comparison.json').write_text(json.dumps(rows, indent=2))
''')
md('''
## Export model và lưu artifacts

Adapter không phải model đầy đủ để truyền trực tiếp cho evaluator hiện tại. Nếu cần, đặt EXPORT_MODE
rồi merge trên CPU; cần RAM cho model float32 và dung lượng đĩa. Export từng nhánh khi dùng.
Sau đó dùng đường dẫn `merged-*` làm `--model_name_or_path` trong pipeline đánh giá đã audit.
Evaluator cũ chưa có chat/non-thinking wrapper và chưa xác nhận giao thức v2; không dùng điểm của nó
để kết luận cải thiện theo paper trước khi kiểm chứng. Notebook này chưa tích hợp SentiBench evaluator v2.

Save Version với outputs trên Kaggle. File ZIP nhỏ gồm log/manifest/split IDs/config/adapter, không gồm
base model, raw corpus, tokenized splits hay optimizer. Resume cần giữ cả WORK bằng Kaggle outputs.
Trước khi xóa session, kiểm tra output thực sự được lưu. Không có đảm bảo session đủ cho full run.
''')
code('''
if EXPORT_MODE:
    if EXPORT_MODE not in MODES:
        raise ValueError('EXPORT_MODE must be a trained mode')
    run(['export', '--run', WORK / EXPORT_MODE, '--output', WORK / f'merged-{EXPORT_MODE}'],
        log=WORK / f'export-{EXPORT_MODE}.log')
''')
md('''
## Đánh giá So sánh Model gốc (Before) vs Model sau Finetuning (After)

Cell sau tự động chạy đánh giá trên bài kiểm thử chuẩn (mặc định `sc/imdb`, `sc/sst2`, 4-shot)
để đo lường mức độ cải thiện trực tiếp của LoRA adapter so với mô hình gốc trước khi finetune.
Kết quả được xuất ra dạng bảng đối chứng Before vs After và lưu tại `evaluation_comparison.json`.
''')
code('''
# Đánh giá Before vs After
EVAL_DATASETS = 'sc/imdb,sc/sst2'
EVAL_BATCH_SIZE = 4
EVAL_K_SHOT = 4
ADAPTER_PATH = WORK / 'uniform' / 'adapter'

eval_script = BASE / 'source/kaggle_evaluate.py'
if ADAPTER_PATH.exists() and eval_script.exists():
    eval_cmd = [
        sys.executable, str(eval_script),
        '--model', MODEL,
        '--adapter', str(ADAPTER_PATH),
        '--datasets', EVAL_DATASETS,
        '--k_shot', str(EVAL_K_SHOT),
        '--eval_batch_size', str(EVAL_BATCH_SIZE),
        '--output_dir', str(WORK / 'eval_comparison')
    ]
    print("=== Đang chạy đánh giá so sánh Trước vs Sau Finetuning ===")
    subprocess.run(eval_cmd, check=True)
else:
    print(f"Adapter chưa sẵn sàng tại {ADAPTER_PATH}; hãy hoàn tất bước train trước.")
''')
code('''
# Portable run metadata and adapters; full checkpoints stay in WORK for Save Version.
import zipfile
archive = BASE / f'{RUN_TAG}-{MODEL_TAG}-artifacts.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in WORK.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(WORK)
        if any(p.startswith('checkpoint-') or p.startswith('merged-') for p in relative.parts):
            continue
        if relative.parts[0] == 'prepared' and path.name not in {
            'manifest.json', 'preview.json', 'rejected.json'}:
            continue
        z.write(path, relative)
    for name in ('train', 'probe', 'holdout'):
        ids = []
        with (PREPARED / f'{name}.jsonl').open() as stream:
            for line in stream:
                r = json.loads(line)
                ids.append({k: r[k] for k in ('id', 'source_key', 'group', 'train_group')})
        z.writestr(f'prepared/{name}-ids.json', json.dumps(ids))
    for relative, source in BUNDLED.items():
        z.writestr('source/' + relative, source)
print('Artifacts:', archive)
print('Full resume checkpoints:', WORK)
''')
md('''
Nguồn: [corpus v2](https://huggingface.co/datasets/zhang-yice/sentiment-distillation-v2),
[CompEffDist](https://arxiv.org/html/2510.24425v1),
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
Phương pháp progress-weighting là ứng viên nghiên cứu; chưa chứng nhận novelty hay lợi ích.
''')
notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
    'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'distill-{i:03d}'
path = ROOT / 'notebooks/Sentiment_Distillation_Train_Kaggle.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(path)
