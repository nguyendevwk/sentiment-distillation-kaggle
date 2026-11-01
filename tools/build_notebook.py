"""Build a notebook with bundled source and optional authenticated private clone."""
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

Notebook hỗ trợ clone repo private bằng GitHub token hoặc sử dụng source đã đóng gói.
Upload duy nhất file `.ipynb` này lên Kaggle, bật **GPU** và **Internet**, chạy lần lượt các cell.
Colab: chọn runtime GPU. Đặt `USE_PRIVATE_REPO=False` nếu muốn dùng bản đóng gói, không cần GitHub token.

Mặc định: Qwen 0.5B, SST2, 5 mẫu đầu, batch 1, 4-shot, 2.048 token đầu vào.
Đây là chạy thử, không phải kết quả toàn benchmark. Không có huấn luyện/fine-tuning trong notebook.

Mặc định dùng trực tiếp thư viện có sẵn của nền tảng (`ENV_MODE='platform'`).
Không cài dependency, không tạo venv và không ép phiên bản Python/thư viện trong chế độ này.
Chỉ ghi lại phiên bản, kiểm tra gói cần thiết có tồn tại và thông tin GPU; không chạy kiểm tra model giả lập.
Nếu muốn dùng bộ phiên bản cố định trong venv, đổi `ENV_MODE='venv'`.
Notebook chưa được chạy thực tế trên Kaggle/Colab; thư viện nền tảng vẫn có thể phát sinh lỗi khi chạy model thật.
''')
code('''
from pathlib import Path
import os, sys, json, subprocess, hashlib, shutil

BASE = Path('/kaggle/working/sentiment-distillation' if Path('/kaggle').exists()
            else '/content/sentiment-distillation' if Path('/content').exists()
            else './sentiment-notebook-run').resolve()
BASE.mkdir(parents=True, exist_ok=True)
USE_PRIVATE_REPO = True
PRIVATE_REPO = 'nguyendevwk/sentiment-distillation-kaggle'
PRIVATE_BRANCH = 'main'
GITHUB_AUTH = 'secrets'  # 'secrets': Kaggle/Colab Secrets; 'prompt': ô nhập token ẩn khi chạy.
PROJECT_DIR = BASE / 'private-source' if USE_PRIVATE_REPO else BASE
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
ENV_MODE = 'platform'  # 'platform': dùng gói có sẵn; 'venv': cài bộ dependency cố định.
# MODEL cũng nhận thư mục model đã attach, ví dụ /kaggle/input/.../model
DATA_DIR = None  # Hoặc đường dẫn chứa sc/, multifaced/, absa/ trong Kaggle Input.
DATASETS = ['sc/sst2']
SEEDS = [42]
MAX_SAMPLES = 5  # 0 = toàn bộ test. Giữ nguyên cấu hình khi tiếp tục checkpoint.
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
ENV_DIR = Path(f'/tmp/sentiment-eval-py{sys.version_info.major}{sys.version_info.minor}')
PYTHON = ENV_DIR / 'bin/python'
print('Workspace:', BASE, '\\nOutput:', OUTPUT)
''')

markdown('''
## GitHub token để clone repo private

**Kaggle:** mở **Add-ons → Secrets**, thêm secret tên `GITHUB_TOKEN`, dán GitHub Personal Access Token
vào **Value** và bật quyền sử dụng secret cho notebook. Tên secret phải đúng là `GITHUB_TOKEN`.
Token fine-grained chỉ cần chọn repo này và cấp **Contents: Read-only** (Metadata đi kèm).
[Tạo token](https://github.com/settings/personal-access-tokens/new).

Hoặc đặt `GITHUB_AUTH='prompt'`: khi chạy sẽ hiện ô nhập token ẩn. Không dán token vào code/config.
Colab có thể dùng Secrets cùng tên (cấp Notebook access), hoặc chọn `prompt`.

Cell clone vào `private-source/`; chạy lại sẽ fetch và cập nhật nhánh bằng fast-forward.
Nếu checkout có file đã sửa, cell dừng để tránh ghi đè. Token chỉ truyền tạm cho tiến trình Git,
không lưu vào URL remote, file credentials, notebook output hoặc log.
Chọn `USE_PRIVATE_REPO=False` để bỏ qua cell này và dùng source đóng gói.
''')
code(r'''
def clone_private_source():
    import getpass, re, tempfile
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', PRIVATE_REPO):
        raise ValueError('PRIVATE_REPO phải có dạng owner/repo, không chứa token hoặc URL.')
    if not PRIVATE_BRANCH or PRIVATE_BRANCH.startswith('-'):
        raise ValueError('PRIVATE_BRANCH không hợp lệ.')
    if GITHUB_AUTH == 'prompt':
        token = getpass.getpass('GitHub token (ẩn, không lưu vào notebook): ').strip()
    elif GITHUB_AUTH == 'secrets':
        try:
            if Path('/kaggle').exists():
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                secret_value_0 = user_secrets.get_secret("GITHUB_TOKEN")
                token = secret_value_0.strip()
                del secret_value_0
            else:
                from google.colab import userdata
                token = (userdata.get("GITHUB_TOKEN") or '').strip()
        except Exception:
            raise RuntimeError('Không đọc được secret. Thêm GITHUB_TOKEN vào Secrets và cấp quyền cho notebook, '
                               "hoặc đặt GITHUB_AUTH='prompt'.") from None
    else:
        raise ValueError("GITHUB_AUTH chỉ nhận 'secrets' hoặc 'prompt'.")
    if not token:
        raise ValueError('GitHub token trống.')
    url = f'https://github.com/{PRIVATE_REPO}.git'
    git_env = os.environ.copy()
    git_env.update(PRIVATE_CLONE_TOKEN=token, GIT_TERMINAL_PROMPT='0',
                   GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    # Helper contains no token; the token exists only in the Git subprocess environment.
    with tempfile.TemporaryDirectory(prefix='sentiment-git-auth-') as auth_dir:
        askpass = Path(auth_dir) / 'askpass.sh'
        askpass.write_text(r"""#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *) printf '%s\n' "$PRIVATE_CLONE_TOKEN" ;;
esac
""", encoding='utf-8')
        askpass.chmod(0o700)
        git_env['GIT_ASKPASS'] = str(askpass)

        def git(arguments, cwd=None):
            result = subprocess.run(['git', '-c', 'credential.helper=', *arguments], cwd=cwd,
                                    env=git_env, capture_output=True, text=True)
            if result.returncode:
                # No raw stderr/exception: authentication errors must never reveal credentials.
                raise RuntimeError('Git thất bại (không in chi tiết xác thực). Kiểm tra Internet, repo/branch, '
                                   'token còn hạn và quyền Contents: Read-only; checkout phải sạch.') from None
            return result.stdout.strip()

        if (PROJECT_DIR / '.git').is_dir():
            if git(['remote', 'get-url', 'origin'], PROJECT_DIR) != url:
                raise RuntimeError('Checkout hiện tại thuộc repo khác. Chọn workspace khác.')
            if git(['status', '--porcelain'], PROJECT_DIR):
                raise RuntimeError('Checkout có thay đổi cục bộ. Lưu thay đổi hoặc chọn workspace khác trước khi cập nhật.')
            if git(['branch', '--show-current'], PROJECT_DIR) != PRIVATE_BRANCH:
                raise RuntimeError('Checkout đang ở nhánh khác. Đổi PRIVATE_BRANCH hoặc chọn workspace khác.')
            shallow = git(['rev-parse', '--is-shallow-repository'], PROJECT_DIR) == 'true'
            git(['fetch', *(['--unshallow'] if shallow else []), 'origin', PRIVATE_BRANCH], PROJECT_DIR)
            git(['merge', '--ff-only', 'FETCH_HEAD'], PROJECT_DIR)
        else:
            if PROJECT_DIR.exists() and any(PROJECT_DIR.iterdir()):
                raise RuntimeError('Thư mục clone đã tồn tại và không rỗng. Chọn workspace khác.')
            git(['clone', '--depth', '1', '--branch', PRIVATE_BRANCH, '--single-branch', url, str(PROJECT_DIR)])
        commit = git(['rev-parse', 'HEAD'], PROJECT_DIR)
    git_env.pop('PRIVATE_CLONE_TOKEN', None)
    token = None
    return commit

if USE_PRIVATE_REPO:
    source_commit = clone_private_source()
    print('Private source ready:', PRIVATE_REPO, 'commit', source_commit)
else:
    print('Using bundled source; GitHub token not needed.')
''')

files = [ROOT / name for name in ('fsa.py', 'icl_multifaced.py', 'utils.py', 'parse.py',
         'notebook_runtime.py', 'check_notebook_environment.py', 'requirements-notebook.txt',
         'requirements-notebook-py313.txt', 'model_name.json')]
files += sorted((ROOT / 'parse_utils').glob('*.py'))
sources = {str(p.relative_to(ROOT)): p.read_text() for p in files}
markdown('''
## Source đi kèm
Cell sau chỉ ghi source đóng gói khi `USE_PRIVATE_REPO=False`; không ghi đè source clone từ repo private.
Các nội dung file được đóng gói trực tiếp bên dưới để notebook hoạt động độc lập.
''')
code('PATCHED_SOURCES = ' + repr(sources) + '''
for relative, content in PATCHED_SOURCES.items():
    if USE_PRIVATE_REPO:
        break
    destination = BASE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')
print('Source:', PROJECT_DIR)
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
    DATA_DIR = PROJECT_DIR / 'datasets'
    marker = DATA_DIR / '.source-commit'
    if not USE_PRIVATE_REPO and (not marker.exists() or marker.read_text() != UPSTREAM_COMMIT):
        archive = BASE / 'upstream-data.zip'
        url = f'https://codeload.github.com/HITSZ-HLT/Sentiment-Distillation/zip/{UPSTREAM_COMMIT}'
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                parts = Path(item.filename).parts
                if len(parts) < 3 or parts[1] != 'datasets' or item.is_dir():
                    continue
                destination = (PROJECT_DIR / Path(*parts[1:])).resolve()
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
## Chọn môi trường chạy
Mặc định `platform`: dùng Python và các gói sẵn có, không gọi pip/venv/ensurepip, không ép phiên bản.
Tùy chọn `venv`: bộ dependency riêng cho Python 3.10–3.13, giữ PyTorch/CUDA hệ thống.
Ở chế độ venv, cài wheel, xem kế hoạch pip trước khi cài và dừng nếu định thay PyTorch/CUDA.
''')
code('''
ENVIRONMENT_OK = False
PROCESS_ENV = os.environ.copy()
PROCESS_ENV.pop('PYTHONPATH', None)
PROCESS_ENV.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')

def run_logged(command, filename, cwd=PROJECT_DIR):
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
if ENV_MODE == 'platform':
    PYTHON = Path(sys.executable)
    print('Platform mode: using', PYTHON, '; no installation or strict version checks.')
elif ENV_MODE == 'venv':
    PYTHON = ENV_DIR / 'bin/python'
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError(f'Python {sys.version_info[:2]} chưa nằm trong cấu hình dependency này (3.10–3.13).')
    REQUIREMENTS_PATH = PROJECT_DIR / ('requirements-notebook-py313.txt' if sys.version_info[:2] == (3, 13)
                                       else 'requirements-notebook.txt')
    BNB_VERSION = '0.46.1' if sys.version_info[:2] == (3, 13) else '0.45.5'
    PROCESS_ENV['PYTHONNOUSERSITE'] = '1'
    host = json.loads(subprocess.check_output([sys.executable, '-c', probe], env=PROCESS_ENV, text=True))
    if not host['gpu']:
        raise RuntimeError('Chưa có CUDA GPU. Bật GPU trong Settings/Runtime rồi chạy lại.')
    (OUTPUT / 'host-environment.json').write_text(json.dumps(host, indent=2))
    run_logged([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', ENV_DIR],
               'venv-create.log')
    run_logged([sys.executable, '-m', 'pip', '--python', PYTHON, 'install',
                '--ignore-installed', '--no-deps', 'pip==25.3'], 'pip-bootstrap.log')
    constraints = BASE / 'host-torch-constraints.txt'
    constraints.write_text('\\n'.join(f'{name}=={v}' for name, v in host['constraints'].items()) + '\\n')
    install = [PYTHON, '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed',
               '--only-binary=:all:', '--constraint', constraints, '-r', REQUIREMENTS_PATH]
    if FOUR_BIT:
        install.append(f'bitsandbytes=={BNB_VERSION}')
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
else:
    raise ValueError("ENV_MODE phải là 'platform' hoặc 'venv'.")
''')
markdown('''
## Thông tin môi trường
`platform`: ghi phiên bản/gói thiếu và GPU; bỏ kiểm tra pin và model giả lập. Không tự cài gói thiếu.
`venv`: kiểm tra dependency đầy đủ, import và thử model nhỏ; nếu bật 4-bit, thử bitsandbytes.
Kết quả kiểm tra không đảm bảo model thật đủ VRAM hoặc mọi API tương thích.
''')
code('''
check = [PYTHON, PROJECT_DIR / 'check_notebook_environment.py',
         '--report', OUTPUT / 'environment-report.json']
if ENV_MODE == 'platform':
    check.append('--platform')
else:
    check.extend(['--require-gpu', '--requirements', REQUIREMENTS_PATH])
if FOUR_BIT:
    check.append('--four-bit')
log = run_logged(check, 'environment-check.log')
print(log.read_text()[-10000:])
if ENV_MODE == 'venv':
    run_logged([PYTHON, '-m', 'pip', 'freeze'], 'effective-packages.txt')
ENVIRONMENT_OK = True
''')
markdown('''
## Chạy đánh giá / tiếp tục tiến độ
Mặc định chạy thử 5 mẫu SST2 với Qwen 0.5B. Đổi `MAX_SAMPLES=0` để chạy đầy đủ một dataset.
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
        command = [PYTHON, PROJECT_DIR / script, '--dataset', dataset, '--subname', EXPERIMENT,
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
summary = subprocess.run([str(PYTHON), str(PROJECT_DIR / 'parse.py'), '--path', str(OUTPUT / 'result.txt')],
                         check=True, env=PROCESS_ENV, cwd=PROJECT_DIR, capture_output=True, text=True)
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
