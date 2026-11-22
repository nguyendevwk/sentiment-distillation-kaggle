"""Build a standalone Kaggle A/B notebook; never download or execute models."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(value):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(value).strip() + '\n'})


def code(value):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(value).strip() + '\n',
                  'execution_count': None, 'outputs': []})


md('''
# Distillation sentiment v1 với hai phương án A và B

**A:** giữ độ phủ + chọn KNOWDIST theo tỷ số response loss của student.
**B:** giữ độ phủ, không cần chấm model, luôn chuẩn bị trước để dự phòng.
**R:** KNOWDIST random làm đối chứng. ICLDIST và dev giống nhau giữa cả ba.

Đây là pipeline cần kiểm chứng GPU, không có bảo đảm tăng F1 hoặc tính mới.
Mặc định chỉ chuẩn bị B/R và chạy dry-run; chưa train cho đến khi RUN_TRAIN=True.
Chuyển ACTIVE_PLAN mà giữ thư mục riêng; không nối K của A sang I của B.
Lần đầu chạy theo cell và đọc preview. Sau khi có checkpoint, chạy lại bootstrap,
config, runtime và cell train; không chạy lại các cell tạo output không rỗng.
Lưu full outputs bằng Save Version. ZIP adapter không đủ resume.
''')

sources = {f'training/{name}': (ROOT / 'training' / name).read_text() for name in
           ('distill_core.py', 'v1_core.py', 'train_distill.py', 'prepare_v1_pruned.py', 'run_v1_ab.py')}
sources['configs/v1_ab.json'] = (ROOT / 'configs/v1_ab.json').read_text()
code('''
from pathlib import Path
import json, os, sys, subprocess, importlib.metadata
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
BASE = Path('/kaggle/working/sentiment-v1-ab')
PROJECT = BASE / 'source'
PROJECT.mkdir(parents=True, exist_ok=True)
'''+ '\nSOURCES = ' + repr(sources) + '''
for name, content in SOURCES.items():
    path = PROJECT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
print('Embedded source:', PROJECT)
''')

code('''
EXPERIMENT = 'qwen3-06b-bctd-01'  # New tag when settings/source change
MODEL = 'Qwen/Qwen3-0.6B'  # Or 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = 'main'  # tokenizer resolves and records immutable SHA
DATASET_REVISION = 'main'  # download resolves SHA before normalization
LOCAL_CORPUS = None  # Or attached raw v1 directory under /kaggle/input
LOCAL_CORPUS_REVISION = None  # Exact upstream revision, required for LOCAL_CORPUS
BENCHMARK_ROOT = None  # Attached SentiBench root; exact-overlap audit if provided
MAPPING_CONFIRMED = True  # Confirmed: inspect raw records and extractor first
ACTIVE_PLAN = 'BCTD'  # Options: 'BCTD', 'B0', 'D', 'F', 'B', 'R', 'A'
RUN_SCORE_A = False
RUN_TRAIN = True  # True to execute training, False for dry-run
PILOT_STEPS = 50  # 0 = token-budgeted main run; >0 for pilot verification
SEED = 42
STAGE1_FORK = None  # Path to existing knowdist adapter to reuse Stage 1
WORK = BASE / EXPERIMENT
NORMALIZED = WORK / 'normalized'
POOL = WORK / 'pool'
SCORES = WORK / 'scores-A'
PREPARED = WORK / 'prepared'
RUNS = WORK / ('pilot-runs' if PILOT_STEPS else 'main-runs')
WORK.mkdir(parents=True, exist_ok=True)
def run(module, *args, check=True):
    cmd = [sys.executable, str(PROJECT / 'training' / module), *map(str, args)]
    print(' '.join(cmd))
    return subprocess.run(cmd, check=check)
''')

md('''
## Kiểm tra môi trường thực tế

Không tự thay torch/CUDA hoặc hạ phiên bản thư viện. Thiếu package thì dừng và
thiết lập môi trường Kaggle phù hợp trước khi chạy. Chỉ kiểm tra này chưa chứng
minh mọi import/model operation tương thích. Bật GPU khi tokenize/scoring/train;
normalize có thể chạy trong một notebook CPU riêng để tiết kiệm quota.
''')
code('''
packages = ['torch', 'transformers', 'peft', 'accelerate', 'huggingface-hub', 'numpy', 'safetensors']
versions = {p: importlib.metadata.version(p) for p in packages}
import torch
environment = {'python': sys.version, 'packages': versions,
    'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda,
    'gpus': [{'name': torch.cuda.get_device_name(i), 'capability': torch.cuda.get_device_capability(i),
              'total_bytes': torch.cuda.get_device_properties(i).total_memory}
             for i in range(torch.cuda.device_count())]}
(WORK / 'environment.json').write_text(json.dumps(environment, indent=2))
print(json.dumps(environment, indent=2))
''')

md('''
## Tải v1 và kiểm tra schema

Không tải teacher/model 70B. Nếu dữ liệu đã attach, điền revision chính xác.
Xem các trường prompt/response và cách lấy văn bản cuối trong ICLDIST. Các template
ngoài các marker được hỗ trợ sẽ bị loại có lý do. Không coi phần bị loại là nhiễu
nếu chỉ chưa có extractor. Chỉ xác nhận mapping sau khi kiểm tra.
''')
code('''
if LOCAL_CORPUS:
    if not LOCAL_CORPUS_REVISION:
        raise ValueError('Provide actual LOCAL_CORPUS_REVISION')
    CORPUS, CORPUS_SHA = Path(LOCAL_CORPUS), LOCAL_CORPUS_REVISION
else:
    CORPUS = WORK / 'raw-v1'
    run('prepare_v1_pruned.py', 'download', '--revision', DATASET_REVISION, '--output', CORPUS)
    CORPUS_SHA = json.loads((CORPUS / 'download-manifest.json').read_text())['revision']
sys.path.insert(0, str(PROJECT / 'training'))
from v1_core import iter_objects
preview_files = [p for p in sorted(CORPUS.rglob('*')) if p.suffix in ('.json', '.jsonl')
                 and not p.name.endswith('manifest.json') and '.cache' not in p.parts]
for path in preview_files[:60]:
    first = next(iter_objects(path), None)
    print(path.relative_to(CORPUS), json.dumps(first, ensure_ascii=False)[:2200])
''')

code('''
if not MAPPING_CONFIRMED:
    raise ValueError('Inspect raw records, set MAPPING_CONFIRMED=True and rerun this cell')
args = ['normalize', '--input', CORPUS, '--output', NORMALIZED, '--revision', CORPUS_SHA,
        '--mapping-confirmed', '--seed', SEED]
if BENCHMARK_ROOT:
    args += ['--benchmark-root', BENCHMARK_ROOT]
run('prepare_v1_pruned.py', *args)
print((NORMALIZED / 'manifest.json').read_text())
print('Inspect exclusions.jsonl; source grouping does not isolate embedded demo text.')
''')

code('''
run('prepare_v1_pruned.py', 'tokenize', '--normalized', NORMALIZED, '--output', POOL,
    '--model', MODEL, '--model-revision', MODEL_REVISION, '--seed', SEED,
    '--know-candidates', 10000, '--icl-candidates', 5000, '--max-length', 2048,
    '--rpr-filter')
print((POOL / 'manifest.json').read_text())
print((POOL / 'preview.json').read_text())
''')

md('''
## Chuẩn bị dữ liệu cho các nhánh (B, R, B0, D, F, BCTD)

Các nhánh dùng chung ICLDIST và dev. Nhánh D và BCTD kích hoạt Stratified Task-Frequency Allocation (STFA).
Ngân sách tính theo token.
''')
code('''
plans = [('B', 'coverage'), ('R', 'random'), ('B0', 'coverage'), ('D', 'coverage'), ('F', 'coverage'), ('BCTD', 'coverage')]
for plan_name, method in plans:
    cmd = ['select', '--pool', POOL, '--output', PREPARED / plan_name, '--method', method, '--seed', SEED]
    if plan_name in ('D', 'BCTD'):
        cmd += ['--stratified-icl']
    run('prepare_v1_pruned.py', *cmd)
print('Prepared plans:', [p[0] for p in plans])
''')

md('''
## Chấm và chuẩn bị A nếu đủ ngân sách (Tùy chọn)

Đây là proxy dùng base student, không tái lập nguyên quy trình IFD có học trước.
Khoảng IFD [0,1] là cấu hình thí nghiệm ban đầu, không chứng nhận đáp án đúng.
Score có resume, mỗi lượt tối đa 2h; lỗi/thiếu điểm không ngăn dùng B đã chuẩn bị.
Chi phí chấm phải cộng vào tổng chi phí A. Có thể rerun score với --resume.
''')
code('''
if RUN_SCORE_A:
    args = ['score', '--pool', POOL, '--output', SCORES, '--max-seconds', 7200]
    if (SCORES / 'identity.json').exists():
        args += ['--resume']
    result = run('prepare_v1_pruned.py', *args, check=False)
    summary_file = SCORES / 'summary.json'
    if result.returncode == 0 and summary_file.exists() and json.loads(summary_file.read_text())['completed']:
        result = run('prepare_v1_pruned.py', 'select', '--pool', POOL, '--scores', SCORES,
            '--method', 'student', '--output', PREPARED / 'A', '--seed', SEED, check=False)
        if result.returncode:
            print('A selection failed. Inspect coverage/scores; B remains available. Set ACTIVE_PLAN=B explicitly.')
    else:
        print('A not complete. Preserve score outputs; resume later or use B explicitly.')
else:
    print('A scoring disabled. B, R, B0, D, F, BCTD are ready.')
''')

md('''
## Huấn luyện mô hình (Hai giai đoạn hoặc Fork Stage 1)

Chạy ACTIVE_PLAN đã chọn (mặc định: BCTD). R/B/B0/D/F/BCTD đều có thư mục riêng.
Nếu tái sử dụng Stage 1 adapter từ account/run khác, gán STAGE1_FORK để tiết kiệm 75% thời gian.
''')
code('''
valid_plans = ('A', 'B', 'B0', 'D', 'F', 'BCTD', 'R')
if ACTIVE_PLAN not in valid_plans:
    raise ValueError(f'Unknown plan: {ACTIVE_PLAN}. Must be one of {valid_plans}')
if not (PREPARED / ACTIVE_PLAN / 'selection.json').exists():
    raise ValueError(f'Plan {ACTIVE_PLAN} is not prepared; check PREPARED directory')
args = ['--prepared', PREPARED / ACTIVE_PLAN, '--output', RUNS / ACTIVE_PLAN / f'seed-{SEED}',
        '--plan', ACTIVE_PLAN, '--seed', SEED, '--max-seconds', 28800, '--pilot-steps', PILOT_STEPS]
if STAGE1_FORK and Path(STAGE1_FORK).exists():
    args += ['--fork-stage1', str(STAGE1_FORK)]
if not RUN_TRAIN:
    args += ['--dry-run']
run('run_v1_ab.py', *args)
''')

md('''
## Đánh giá Checkpoint trên SentiBench Validation
''')
code('''
target_adapter = RUNS / ACTIVE_PLAN / f'seed-{SEED}' / 'icldist' / 'adapter'
if target_adapter.exists():
    print('Stage 2 adapter ready:', target_adapter)
    summary_file = RUNS / ACTIVE_PLAN / f'seed-{SEED}' / 'icldist' / 'summary.json'
    if summary_file.exists():
        print(summary_file.read_text())
else:
    print('No adapter found yet at:', target_adapter)
''')

md('''
## Chuyển nhánh và đánh giá

Chọn A/B theo probe/dev và tổng chi phí, không theo benchmark test; holdout chỉ
báo cáo độc lập. Teacher-response CE không phải sentiment F1. Không tự kết luận A
tốt hơn nếu IFD/loss thấp hơn. Cần audited SentiBench evaluator và cùng test/demo
cho mọi checkpoint; ASQP notebook cũ đang cần audit. Không dùng old cache giữa adapters.

Để ablate fine-tuning, dùng một config mới (ví dụ mode=progress cho cả nhánh đối
chứng tương ứng), không sửa mode trong run đang resume. Trước hết giữ uniform
để tách hiệu quả chọn dữ liệu. Chưa có bảo đảm novelty hoặc F1 tăng.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': True}},
    'nbformat': 4, 'nbformat_minor': 5}
# Deterministic cell IDs for nbformat 4.5.
for i, cell in enumerate(cells):
    cell['id'] = f'v1-ab-{i:03d}'
target = ROOT / 'notebooks/Sentiment_Distillation_V1_AB_Kaggle.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(target)
