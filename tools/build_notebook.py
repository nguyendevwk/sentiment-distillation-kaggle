"""Build a standalone notebook containing the patched evaluator (no private GitHub token needed)."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []

def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})

def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n',
                  'execution_count': None, 'outputs': []})

markdown('''
# Sentiment Distillation — Kaggle / Colab

Notebook độc lập: chứa source đánh giá đã chỉnh sửa; chỉ tải **dữ liệu** từ commit công khai của tác giả.
Upload duy nhất file `.ipynb` này lên Kaggle, bật **GPU** và **Internet**, chạy lần lượt các cell.
Colab: chọn runtime GPU. Không cần token GitHub của repo private.

Mặc định: Qwen 1.5B, SST2, 32 mẫu đầu, batch 1, 4-shot, 2.048 token đầu vào.
Đây là chạy thử, không phải kết quả toàn benchmark. Không có huấn luyện/fine-tuning trong notebook.

Môi trường riêng dùng `--system-site-packages` để tái sử dụng PyTorch/CUDA của nền tảng.
Các gói dự án được cài trong venv; các chương trình đánh giá chạy bằng Python của venv,
không dùng kernel để import các thư viện đã thay phiên bản. Đây là cách ly một phần, không phải container.
Có kiểm tra kế hoạch pip, dependency đang dùng, import, tiny-model forward/generation và GPU.
Nếu không đạt, notebook dừng. Chưa được chạy thực tế trên Kaggle; kiểm tra sẽ thực hiện trong phiên của bạn.
''')
code('''
from pathlib import Path
import os, sys, json, subprocess, hashlib, shutil

BASE = Path('/kaggle/working/sentiment-distillation' if Path('/kaggle').exists()
            else '/content/sentiment-distillation' if Path('/content').exists()
            else './sentiment-notebook-run').resolve()
BASE.mkdir(parents=True, exist_ok=True)
MODEL = 'zhang-yice/Qwen2.5-1.5B-sentiment-distillation-v1'
# MODEL cũng nhận thư mục model đã attach, ví dụ /kaggle/input/.../model
DATA_DIR = None  # Hoặc đường dẫn chứa sc/, multifaced/, absa/ trong Kaggle Input.
DATASETS = ['sc/sst2']
SEEDS = [42]
MAX_SAMPLES = 32  # 0 = toàn bộ test. Giữ nguyên cấu hình khi tiếp tục checkpoint.
BATCH_SIZE = 1
K_SHOT = 4
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 150  # Dùng cho FSA; phân loại sử dụng điểm token đầu.
DEMO_SPLIT = 'train'  # Giữ cách lấy demo của source. 'val' cần val.json ở từng dataset.
FOUR_BIT = False  # Chỉ bật nếu cần giảm VRAM; có cài và kiểm tra bitsandbytes riêng.
DTYPE = 'auto'  # BF16 nếu GPU hỗ trợ, FP16 trên GPU khác.
OUTPUT = BASE / 'output'  # Colab: có thể đổi sang thư mục Google Drive đã mount.
OUTPUT.mkdir(parents=True, exist_ok=True)
EXPERIMENT = 'qwen-notebook'
ENV_DIR = Path('/tmp/sentiment-eval-venv')
PYTHON = ENV_DIR / 'bin/python'
print('Workspace:', BASE, '\\nOutput:', OUTPUT)
''')

files = [ROOT / name for name in ('fsa.py', 'icl_multifaced.py', 'utils.py', 'parse.py',
         'notebook_runtime.py', 'check_notebook_environment.py', 'requirements-notebook.txt', 'model_name.json')]
files += sorted((ROOT / 'parse_utils').glob('*.py'))
sources = {str(p.relative_to(ROOT)): p.read_text() for p in files}
markdown('''
## Source đi kèm
Cell sau ghi source đã chỉnh sửa vào workspace. Không clone/chạy source mới từ nhánh upstream.
Các nội dung file được đóng gói trực tiếp bên dưới để notebook hoạt động độc lập.
''')
code('PATCHED_SOURCES = ' + repr(sources) + '''
for relative, content in PATCHED_SOURCES.items():
    destination = BASE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')
print('Materialized', len(PATCHED_SOURCES), 'source/config files')
''')
markdown('''
## Dữ liệu benchmark
Nếu đã attach dữ liệu vào Kaggle, đặt `DATA_DIR` ở cell cấu hình để bỏ qua tải.
Mặc định chỉ tải các file trong `datasets/` từ commit cố định của repo tác giả, không cần đăng nhập GitHub.
''')
code('''
import urllib.request, zipfile
UPSTREAM_COMMIT = '2810fc2826e8f7c2ae6a0d6dd328ab7d8979744a'
if DATA_DIR is None:
    DATA_DIR = BASE / 'datasets'
    marker = DATA_DIR / '.source-commit'
    if not marker.exists() or marker.read_text() != UPSTREAM_COMMIT:
        archive = BASE / 'upstream-data.zip'
        url = f'https://codeload.github.com/HITSZ-HLT/Sentiment-Distillation/zip/{UPSTREAM_COMMIT}'
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                parts = Path(item.filename).parts
                if len(parts) < 3 or parts[1] != 'datasets' or item.is_dir():
                    continue
                destination = (BASE / Path(*parts[1:])).resolve()
                if not destination.is_relative_to(DATA_DIR.resolve()):
                    raise ValueError('Unsafe archive path')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read(item))
        marker.write_text(UPSTREAM_COMMIT)
        archive.unlink()
DATA_DIR = Path(DATA_DIR).resolve()
for dataset in DATASETS:
    for split in ('test', DEMO_SPLIT):
        path = DATA_DIR / dataset / f'{split}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; change DATA_DIR or DEMO_SPLIT')
print('Dataset root:', DATA_DIR)
''')
markdown('''
## Cài dependency mà không thay các gói hệ thống
Hỗ trợ cấu hình Python 3.10–3.12; phiên bản khác sẽ dừng thay vì ép cài wheel không tương thích.
Cell tạo venv, khóa phiên bản PyTorch/CUDA đã có và xem kế hoạch pip trước khi cài.
Nếu kế hoạch định thay torch, triton hoặc các gói NVIDIA, cell sẽ dừng.
Log đầy đủ và kế hoạch cài được lưu trong `output/`.
''')
code('''
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError(f'Python {sys.version_info[:2]} chưa nằm trong cấu hình dependency này (3.10–3.12).')
PROCESS_ENV = os.environ.copy()
PROCESS_ENV.pop('PYTHONPATH', None)
PROCESS_ENV.update(PYTHONNOUSERSITE='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')

def run_logged(command, filename, cwd=BASE):
    log = OUTPUT / filename
    print('Running:', ' '.join(map(str, command[:5])), '... ->', log.name, flush=True)
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), cwd=cwd, env=PROCESS_ENV,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(errors='replace')[-8000:])
        raise RuntimeError(f'Command failed ({result.returncode}); inspect {log}')
    return log

probe = """
import torch, json, importlib.metadata as md
core = {d.metadata['Name']: d.version for d in md.distributions()
        if d.metadata['Name'].lower() in ('torch', 'triton') or d.metadata['Name'].lower().startswith('nvidia-')}
print(json.dumps(dict(torch=torch.__version__, torch_file=torch.__file__, cuda=torch.version.cuda,
                     gpu=torch.cuda.is_available(), constraints=core)))
"""
host = json.loads(subprocess.check_output([sys.executable, '-c', probe], env=PROCESS_ENV, text=True))
if not host['gpu']:
    raise RuntimeError('Chưa có CUDA GPU. Bật GPU trong Settings/Runtime rồi chạy lại.')
(OUTPUT / 'host-environment.json').write_text(json.dumps(host, indent=2))
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(ENV_DIR)], check=True, env=PROCESS_ENV)
run_logged([PYTHON, '-m', 'pip', 'install', 'pip==25.3'], 'pip-bootstrap.log')
constraints = BASE / 'host-torch-constraints.txt'
constraints.write_text('\\n'.join(f'{name}=={v}' for name, v in host['constraints'].items()) + '\\n')
install = [PYTHON, '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed',
           '--constraint', constraints, '-r', BASE / 'requirements-notebook.txt']
if FOUR_BIT:
    install.append('bitsandbytes==0.45.5')
plan = OUTPUT / 'dependency-plan.json'
run_logged(install + ['--dry-run', '--report', plan], 'dependency-resolve.log')
changes = [entry['metadata']['name'].lower().replace('_', '-') for entry in json.loads(plan.read_text())['install']]
blocked = [name for name in changes if name in ('torch', 'triton') or name.startswith('nvidia-')]
if blocked:
    raise RuntimeError(f'Kế hoạch cài định thay CUDA/PyTorch: {blocked}. Dừng để giữ môi trường nền tảng.')
print('Project packages to install:', changes)
run_logged(install, 'dependency-install.log')
inside = json.loads(subprocess.check_output([str(PYTHON), '-c', probe], env=PROCESS_ENV, text=True))
if inside['torch'] != host['torch'] or inside['torch_file'] != host['torch_file']:
    raise RuntimeError('PyTorch trong venv khác bản hệ thống; dừng để kiểm tra.')
print('Dependencies installed in venv; host torch/CUDA retained:', inside['torch'], inside['cuda'])
''')
markdown('''
## Kiểm tra môi trường thực tế
Kiểm tra dependency của những gói evaluator sử dụng, thay vì yêu cầu toàn bộ ứng dụng cài sẵn trên Kaggle tương thích.
Sau đó import thư viện, tạo model nhỏ ngẫu nhiên không cần tải weights và thử forward/generate trên GPU.
Nếu bật 4-bit, thử thêm phép tính bitsandbytes trên GPU. Các kiểm tra này chưa đảm bảo model thật sẽ đủ VRAM.
''')
code('''
check = [PYTHON, BASE / 'check_notebook_environment.py', '--require-gpu',
         '--report', OUTPUT / 'environment-report.json']
if FOUR_BIT:
    check.append('--four-bit')
log = run_logged(check, 'environment-check.log')
print(log.read_text()[-10000:])
# Ghi lại dependency hiệu lực để chẩn đoán/tái lập môi trường.
run_logged([PYTHON, '-m', 'pip', 'freeze'], 'effective-packages.txt')
ENVIRONMENT_OK = True
''')
markdown('''
## Chạy đánh giá / tiếp tục tiến độ
Mặc định chạy thử 32 mẫu SST2. Đổi `MAX_SAMPLES=0` để chạy đầy đủ một dataset.
Để chạy 12 dataset, dùng danh sách ở cell kết quả; nên chia theo dataset và seed để phù hợp thời lượng phiên.
Checkpoint lưu sau từng batch. Chạy lại cùng cấu hình sẽ dùng các dự đoán đã có; cấu hình/data/môi trường khác sẽ bị từ chối.

Giới hạn token có thể cắt bớt instruction/demo và làm thay đổi điểm so với paper; số prompt bị cắt được lưu trong metadata.
Nếu chính câu test dài hơn giới hạn, evaluator dừng để bạn tăng giới hạn thay vì âm thầm bỏ nội dung test.
Không tự đổi precision, số shot hoặc quantization khi hết VRAM vì chúng làm thay đổi điều kiện thực nghiệm.
''')
code('''
if not globals().get('ENVIRONMENT_OK'):
    raise RuntimeError('Chạy cell kiểm tra môi trường thành công trước.')
for dataset in DATASETS:
    for seed in SEEDS:
        configuration = dict(model=MODEL, dataset=dataset, seed=seed, samples=MAX_SAMPLES,
                             batch=BATCH_SIZE, shots=K_SHOT, length=MAX_INPUT_TOKENS,
                             new_tokens=MAX_NEW_TOKENS, demo=DEMO_SPLIT, dtype=DTYPE, four_bit=FOUR_BIT)
        key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
        slug = dataset.replace('/', '-') + f'-seed{seed}-{key}'
        script = 'fsa.py' if dataset.startswith('absa/') else 'icl_multifaced.py'
        command = [PYTHON, BASE / script, '--dataset', dataset, '--subname', EXPERIMENT,
                   '--seed', str(seed), '--model_name_or_path', MODEL,
                   '--data_dir', DATA_DIR, '--output_dir', OUTPUT,
                   '--eval_batch_size', str(BATCH_SIZE), '--k_shot', str(K_SHOT),
                   '--max_seq_length', str(MAX_INPUT_TOKENS), '--max_new_tokens', str(MAX_NEW_TOKENS),
                   '--max_samples', str(MAX_SAMPLES), '--demo_split', DEMO_SPLIT, '--dtype', DTYPE,
                   '--resume_file', OUTPUT / f'{slug}.checkpoint.json']
        if FOUR_BIT:
            command.append('--load_in_4bit')
        print('Evaluating', dataset, 'seed', seed, flush=True)
        log = run_logged(command, f'{slug}.log')
        print(log.read_text(errors='replace')[-2000:])
''')
markdown('''
## Kết quả và lưu lại trước khi kết thúc phiên
`result.txt` chứa điểm; JSON chứa dự đoán; checkpoint dùng để tiếp tục; log/report dùng để báo lỗi.
Dataset chưa chạy được ghi N/A, không tính thành 0; số liệu chạy thử được tách khỏi đánh giá toàn bộ.
Kaggle: Save Version để lưu outputs hoặc tải file ZIP bên dưới. Phiên mới cần attach/copy lại checkpoint vào đúng `OUTPUT`.
Colab: tải ZIP hoặc đổi OUTPUT sang Google Drive đã mount. Dữ liệu trên ổ VM không được đảm bảo sau khi phiên kết thúc.
Không đặt venv/model cache trong file kết quả xuất ra.
''')
code('''
summary = subprocess.run([str(PYTHON), str(BASE / 'parse.py'), '--path', str(OUTPUT / 'result.txt')],
                         check=True, env=PROCESS_ENV, cwd=BASE, capture_output=True, text=True)
print(summary.stdout)
(OUTPUT / 'summary.txt').write_text(summary.stdout)
print('All benchmark datasets:', [
    'sc/imdb', 'sc/yelp2', 'sc/sst2', 'sc/twitter',
    'multifaced/irony18', 'multifaced/tweeteval', 'multifaced/pstance', 'multifaced/intimacy',
    'absa/asqp_rest16', 'absa/opener', 'absa/atsa_rest16', 'absa/acsa_rest16'])
archive = shutil.make_archive(str(BASE.parent / 'sentiment-evaluation-results'), 'zip', OUTPUT)
from IPython.display import FileLink, display
display(FileLink(archive))
''')
markdown('''
## Giới hạn còn lại / khi gửi kết quả cho người bảo trì
- Kaggle/Colab quyết định GPU, quota, RAM và thời lượng phiên; notebook không loại bỏ các giới hạn đó.
- 4-bit chỉ là tùy chọn và có thể đổi điểm; giữ tắt nếu muốn so sánh precision gốc.
- Chạy một GPU, model 1B–1.5B trước; batch 1 vẫn có thể hết VRAM nếu prompt dài.
- Nếu hết VRAM: giảm batch nếu >1; thử 4-bit hoặc giảm shot/độ dài và đặt tên thực nghiệm khác.
- Nếu lỗi dependency, gửi `host-environment.json`, `dependency-resolve.log`, `dependency-install.log`,
  `environment-check.log`, `environment-report.json` (nếu có) và log dataset. Không gửi token.
- Muốn đối chiếu paper: dùng cùng split, prompt, precision và trung bình seed 42/52/62. Source hiện lấy demo
  từ train, còn paper mô tả dev; một số dataset local không có val.json nên cần chuẩn bị split tương ứng.

Nguồn: [Kaggle Notebook docs](https://www.kaggle.com/docs/notebooks),
[Colab FAQ](https://research.google.com/colaboratory/faq.html),
[Paper](https://arxiv.org/html/2503.03225v2),
[Repo tác giả](https://github.com/HITSZ-HLT/Sentiment-Distillation).
''')
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
    'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True}}, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:02d}'
path = ROOT / 'notebooks/Sentiment_Distillation_Kaggle_Colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(path)
