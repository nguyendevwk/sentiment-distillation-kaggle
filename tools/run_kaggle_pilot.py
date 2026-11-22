"""Clone-and-run Kaggle pilot: !python tools/run_kaggle_pilot.py --plan B ...

No local model execution. Outputs are resumable and source/config checked.
Initial benchmark metrics use official validation data, never final test.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from distill_core import digest, file_hash, write_json


def execute(cmd, log):
    """Tee without a shell; retain a failed command and return code."""
    print('\nRUN:', ' '.join(map(str, cmd)), flush=True)
    with Path(log).open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def normalized_output_valid(cmd):
    """Adopt an orphan normalization only when all data/settings are verified."""
    def option(name, default=None):
        return str(cmd[cmd.index(name) + 1]) if name in cmd else default
    directory = Path(option('--output'))
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['corpus_revision'] != option('--revision') or manifest['seed'] != int(option('--seed', '42')):
            return False
        if manifest['dev_fraction'] != float(option('--dev-fraction', '.1')):
            return False
        expected_source = digest([file_hash(ROOT / 'training/prepare_v1_pruned.py'), file_hash(ROOT / 'training/v1_core.py')])
        if manifest['source_sha256'] != expected_source:
            return False
        if file_hash(directory / 'canonical.jsonl') != manifest['canonical_sha256']:
            return False
        raw = Path(option('--input'))
        source_files = sorted(p for p in raw.rglob('*') if p.suffix in ('.json', '.jsonl')
                              and not p.name.endswith('manifest.json') and '.cache' not in p.parts)
        if {str(p.relative_to(raw)): file_hash(p) for p in source_files} != manifest['files']:
            return False
        benchmark = option('--benchmark-root')
        expected = {str(p.relative_to(benchmark)): file_hash(p)
                    for p in sorted(Path(benchmark).glob('*/*/test.json'))} if benchmark else {}
        if manifest['benchmark_files'] != expected:
            return False
        return (directory / 'exclusions.jsonl').is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def recover_normalization(work, cmd):
    directory = Path(str(cmd[cmd.index('--output') + 1]))
    if not directory.exists() or not any(directory.iterdir()):
        return False
    if normalized_output_valid(cmd):
        print('Recovered completed normalization without receipt:', directory, flush=True)
        return True
    archive = work / 'recovery' / f'normalize-incomplete-{time.time_ns()}'
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(archive))
    write_json(archive / 'recovery-reason.json', {'reason': 'Missing receipt and unverifiable/incomplete normalization',
               'original_path': str(directory), 'command': list(map(str, cmd))})
    print('Preserved incomplete normalization:', archive, flush=True)
    return False


def accept_request(work, shared, repair_preparation=False):
    """Allow an explicit source upgrade only before scoring/evaluation/training.

    Retain raw downloads and logs. Archive all derived preparation and old
    receipts together; never mix model runs across a source upgrade.
    """
    request = work / 'pilot-request.json'
    if not request.exists():
        write_json(request, shared)
        return
    previous = json.loads(request.read_text())
    if previous == shared:
        return
    settings = lambda value: {k: v for k, v in value.items() if k != 'source_sha256'}
    if not repair_preparation or settings(previous) != settings(shared):
        raise ValueError('Pilot settings/source differ. For a source-only fix before model work, use --repair-preparation; otherwise choose a new --work.')
    protected = ('runs', 'evaluation', 'scores-A')
    if any(p.is_file() for name in protected for p in (work / name).rglob('*')):
        raise ValueError('Cannot repair preparation after scoring/evaluation/training artifacts exist. Keep this run and use a new --work.')
    archive = work / 'recovery' / f'source-upgrade-{time.time_ns()}'
    archive.mkdir(parents=True)
    shutil.move(str(request), str(archive / request.name))
    for name in ('normalized', 'pool', 'prepared', 'state'):
        path = work / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
    for pattern in ('report-*.json', 'report-*.md', 'failure-*.json'):
        for path in work.glob(pattern):
            shutil.move(str(path), str(archive / path.name))
    (work / 'state').mkdir(exist_ok=True)
    write_json(archive / 'recovery-reason.json', {'reason': 'Explicit preparation-only source upgrade',
               'previous_source_sha256': previous.get('source_sha256'), 'new_source_sha256': shared.get('source_sha256'),
               'preserved_in_place': ['raw-v1', 'logs']})
    write_json(request, shared)
    print('Archived prior preparation; reusing raw download:', archive, flush=True)


def once(work, name, cmd, outputs):
    receipt = work / 'state' / f'{name}.json'
    identity = list(map(str, cmd))
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['command'] != identity:
            raise ValueError(f'{name}: settings changed; use a new --work')
        for path, sha in saved['outputs'].items():
            if not Path(path).is_file() or file_hash(path) != sha:
                raise ValueError(f'{name}: output changed/missing: {path}')
        print('Reuse:', name, flush=True)
        return
    recovered = name == 'normalize' and recover_normalization(work, cmd)
    if not recovered:
        execute(cmd, work / 'logs' / f'{name}.log')
    write_json(receipt, {'command': identity,
                        'outputs': {str(p): file_hash(p) for p in outputs}})


def environment(work):
    required = ['torch', 'transformers', 'tokenizers', 'peft', 'accelerate', 'huggingface-hub',
                'numpy', 'safetensors', 'scikit-learn', 'tqdm']
    packages, missing = {}, []
    for name in required:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            missing.append(name)
    info = {'python': sys.version, 'packages': packages, 'missing': missing,
            'captured_at': datetime.now(timezone.utc).isoformat()}
    if 'torch' not in missing:
        import torch
        info.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
            gpus=[{'name': torch.cuda.get_device_name(i), 'capability': torch.cuda.get_device_capability(i),
                   'total_bytes': torch.cuda.get_device_properties(i).total_memory}
                  for i in range(torch.cuda.device_count())])
    write_json(work / 'environment.json', info)
    write_json(work / 'logs' / f'environment-{time.time_ns()}.json', info)
    if missing:
        raise ValueError(f'Missing Kaggle packages: {missing}. Install compatible packages on Kaggle; torch/CUDA were not changed.')
    if not info.get('cuda_available'):
        raise ValueError('Enable a GPU in Kaggle notebook settings before pilot')
    return info


def eval_command(model, revision, adapter, dataset, output, args):
    cmd = [sys.executable, str(ROOT / 'icl_multifaced_kaggle.py'), '--dataset', dataset,
        '--subname', 'pilot_' + output.parent.name, '--seed', str(args.seed),
        '--model_name_or_path', model, '--model_revision', revision,
        '--data_dir', str(ROOT / 'datasets'), '--output_dir', str(output),
        '--eval_split', 'val', '--demo_split', 'train', '--sample_strategy', 'random',
        '--max_samples', str(args.eval_samples), '--eval_batch_size', '1',
        '--max_seq_length', str(args.max_length), '--k_shot', '4', '--measure_inference']
    if adapter:
        cmd += ['--adapter', str(adapter)]
    return cmd


def collect_report(work, args, wall_seconds):
    report = {'plan': args.plan, 'seed': args.seed, 'pilot_steps_per_stage': args.steps,
              'eval_split': 'val', 'eval_samples_requested': args.eval_samples,
              'invocation_wall_seconds': wall_seconds, 'evaluations': {}, 'training': {},
              'limitations': ['Small validation pilot, not final test or paper-equivalent scores.',
                             'Teacher-response CE is not sentiment F1.',
                             'ASQP excluded pending evaluator audit.',
                             'No installation/download/train/inference was run on the local development machine.']}
    for result_file in sorted((work / 'evaluation').glob('*/*/result.txt')):
        target, dataset_name = result_file.parent.parent.name, result_file.parent.name
        records = [json.loads(line) for line in result_file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not records:
            continue
        report['evaluations'].setdefault(target, {})[dataset_name] = records[-1][1]
    run_root = work / 'runs' / f'pilot-{args.steps}' / args.plan / f'seed-{args.seed}'
    for stage in ('knowdist', 'icldist'):
        summary_path = run_root / stage / 'summary.json'
        if summary_path.exists():
            report['training'][stage] = json.loads(summary_path.read_text())
    score_summary = work / 'scores-A' / 'summary.json'
    if score_summary.exists():
        report['score_A'] = json.loads(score_summary.read_text())
    write_json(work / f'report-{args.plan}.json', report)
    lines = [f'# Kết quả pilot Plan {args.plan}', '',
             'Chấm trên validation, một seed; không phải kết quả test cuối.', '',
             '| Model/checkpoint | Dataset | Macro F1 % | Accuracy % | Forward ms/mẫu | Peak allocated GiB |',
             '|---|---|---:|---:|---:|---:|']
    for target, datasets in report['evaluations'].items():
        for dataset, metrics in datasets.items():
            timing = metrics.get('metadata', {}).get('inference_measurement', {})
            latency = timing.get('amortized_ms_per_example')
            memory = timing.get('peak_allocated_bytes')
            lines.append(f'| {target} | {dataset} | {metrics["f1"] * 100:.2f} | {metrics["acc"] * 100:.2f} | '
                         f'{latency:.2f} | {memory / 1024**3:.3f} |' if latency is not None and memory is not None else
                         f'| {target} | {dataset} | {metrics["f1"] * 100:.2f} | {metrics["acc"] * 100:.2f} | N/A | N/A |')
    lines += ['', 'Forward timing: batch=1, sau warm-up; không gồm nạp model/tokenization/host-to-device/scoring.', '',
              '| Stage | Hoàn tất | Train tokens | Response tokens | Updates thành công | Wall giây* | Peak allocated GiB |',
              '|---|---|---:|---:|---:|---:|---:|']
    for stage, summary in report['training'].items():
        state = summary['state']
        lines.append(f'| {stage} | {summary.get("completed")} | {state["train_tokens"]} | {state["response_tokens"]} | '
                     f'{state["optimizer_updates"]} | {state["elapsed_seconds"]:.2f} | {state["peak_vram_bytes"] / 1024**3:.3f} |')
    lines += ['', '*Train wall gồm probe/holdout/checkpoint trong vòng train; model loading/final export nằm ngoài cửa sổ này.',
              'Giữ log và manifest để giải thích kết quả. Không chọn A/B theo benchmark test.']
    (work / f'report-{args.plan}.md').write_text('\n'.join(lines) + '\n')
    print('\nREPORT:', work / f'report-{args.plan}.md', flush=True)
    return report


def pipeline(args):
    if args.steps < 1 or args.eval_samples < 1 or args.max_length < 3 or not 0 < args.max_seconds <= 36000:
        raise ValueError('Invalid pilot limits')
    work = Path(args.work).resolve()
    if args.dry_run:
        print(json.dumps({'work': str(work), 'plan': args.plan, 'steps_per_stage': args.steps,
            'model': args.model, 'eval_split': 'val', 'eval_samples': args.eval_samples,
            'workflow': ['environment', 'download pinned v1', 'normalize/audit', 'tokenize', 'prepare B/R',
                         'optional score/prepare A', 'before validation evaluation', 'K → I pilot',
                         'after validation evaluation', 'JSON/Markdown report'],
            'note': 'No writes, network, package install or model operations in dry-run.'}, indent=2))
        return
    if not Path('/kaggle/working').is_dir():
        raise ValueError('Execute this launcher on Kaggle; local model execution disabled. --dry-run is available locally.')
    if not work.is_relative_to(Path('/kaggle/working').resolve()):
        raise ValueError('--work must be under writable /kaggle/working')
    if not args.mapping_confirmed:
        raise ValueError('Inspect v1 schema/source extraction, then pass --mapping-confirmed (see training/V1_AB.md).')
    work.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'state'):
        (work / name).mkdir(exist_ok=True)
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in
               [*sorted((ROOT / 'training').glob('*.py')), Path(__file__), ROOT / 'notebook_runtime.py',
                ROOT / 'icl_multifaced_kaggle.py', ROOT / 'utils.py', ROOT / 'configs/v1_ab.json']}
    dataset_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / 'datasets').glob('*/*/*.json'))}
    shared = {'source_sha256': digest(sources), 'dataset_sha256': digest(dataset_hashes),
        'model': args.model, 'model_revision_requested': args.model_revision,
        'dataset_revision_requested': args.dataset_revision, 'local_corpus': args.local_corpus,
        'local_corpus_revision': args.local_corpus_revision, 'seed': args.seed, 'max_length': args.max_length,
        'eval_samples': args.eval_samples, 'steps': args.steps}
    accept_request(work, shared, args.repair_preparation)
    begin = time.monotonic()
    try:
        environment(work)
        prepare = [sys.executable, str(ROOT / 'training/prepare_v1_pruned.py')]
        normalized, pool = work / 'normalized', work / 'pool'
        if args.local_corpus:
            if not args.local_corpus_revision:
                raise ValueError('--local-corpus-revision is required')
            corpus, revision = Path(args.local_corpus), args.local_corpus_revision
        else:
            corpus = work / 'raw-v1'
            # Pin once; reuse this snapshot on rerun even if upstream main changes.
            download_manifest = corpus / 'download-manifest.json'
            if not download_manifest.exists():
                execute(prepare + ['download', '--revision', args.dataset_revision, '--output', corpus],
                        work / 'logs/download.log')
            revision = json.loads(download_manifest.read_text())['revision']
        once(work, 'normalize', prepare + ['normalize', '--input', str(corpus), '--revision', revision,
             '--output', str(normalized), '--mapping-confirmed', '--seed', str(args.seed),
             '--benchmark-root', str(ROOT / 'datasets')], [normalized / 'manifest.json', normalized / 'canonical.jsonl'])
        normalization = json.loads((normalized / 'manifest.json').read_text())
        retained = sum(v for k, v in normalization['counts'].items() if k != 'excluded')
        excluded = normalization['counts'].get('excluded', 0)
        if excluded / max(1, retained + excluded) > .3:
            raise ValueError('More than 30% excluded: inspect exclusions/mapping before training. No automatic data repair.')
        once(work, 'tokenize', prepare + ['tokenize', '--normalized', str(normalized), '--output', str(pool),
             '--model', args.model, '--model-revision', args.model_revision, '--max-length', str(args.max_length),
             '--seed', str(args.seed)], [pool / 'manifest.json', pool / 'pool.jsonl'])
        plans_to_prep = [('B', 'coverage'), ('R', 'random')]
        if args.plan in ('B0', 'D', 'F', 'BCTD'):
            plans_to_prep.append((args.plan, 'coverage'))
        for plan_name, method in plans_to_prep:
            directory = work / 'prepared' / plan_name
            cmd = prepare + ['select', '--pool', str(pool), '--output', str(directory),
                             '--method', method, '--seed', str(args.seed)]
            if plan_name in ('D', 'BCTD'):
                cmd += ['--stratified-icl']
            once(work, 'select-' + plan_name, cmd, [directory / 'selection.json',
                   directory / 'knowdist/manifest.json', directory / 'icldist/manifest.json'])
        if args.plan == 'A':
            scores = work / 'scores-A'
            summary_path = scores / 'summary.json'
            if not summary_path.exists() or not json.loads(summary_path.read_text())['completed']:
                remaining = min(args.score_seconds, args.max_seconds - (time.monotonic() - begin) - 120)
                if remaining <= 0:
                    print('No scoring budget left; rerun to resume, or choose B explicitly.', flush=True)
                    return
                score_cmd = prepare + ['score', '--pool', str(pool), '--output', str(scores), '--max-seconds', str(remaining)]
                if (scores / 'identity.json').exists():
                    score_cmd += ['--resume']
                execute(score_cmd, work / 'logs/score-A.log')
            if not json.loads((scores / 'summary.json').read_text())['completed']:
                print('A scoring incomplete. Rerun to resume, or change --plan B using the same --work.', flush=True)
                return
            directory = work / 'prepared/A'
            once(work, 'select-A', prepare + ['select', '--pool', str(pool), '--output', str(directory),
                 '--method', 'student', '--scores', str(scores), '--seed', str(args.seed)],
                 [directory / 'selection.json', directory / 'knowdist/manifest.json', directory / 'icldist/manifest.json'])
        manifest = json.loads((pool / 'manifest.json').read_text())
        targets = [('before', None)]
        if args.include_paper_reference:
            targets.append(('paper', 'reference'))
        for label, adapter in targets:
            model = 'zhang-yice/Qwen2.5-1.5B-sentiment-distillation-v1' if adapter == 'reference' else args.model
            if adapter == 'reference':
                from huggingface_hub import HfApi
                pin = work / 'paper-revision.json'
                if not pin.exists():
                    write_json(pin, {'model': model, 'revision': HfApi().model_info(model).sha})
                model_revision = json.loads(pin.read_text())['revision']
            else:
                model_revision = manifest['model_revision']
            for dataset in ('sc/sst2', 'multifaced/tweeteval'):
                directory = work / 'evaluation' / label / dataset.replace('/', '_')
                directory.mkdir(parents=True, exist_ok=True)
                once(work, 'eval-' + label + '-' + dataset.replace('/', '_'),
                     eval_command(model, model_revision, None, dataset, directory, args), [directory / 'result.txt'])
        remaining = args.max_seconds - (time.monotonic() - begin)
        if remaining <= 180:
            print('Preparation/evaluation used this invocation budget. Rerun same command to resume.', flush=True)
            return
        run_dir = work / 'runs' / f'pilot-{args.steps}' / args.plan / f'seed-{args.seed}'
        cmd = [sys.executable, str(ROOT / 'training/run_v1_ab.py'), '--prepared', str(work / 'prepared' / args.plan),
               '--output', str(run_dir), '--plan', args.plan, '--seed', str(args.seed),
               '--pilot-steps', str(args.steps), '--max-seconds', str(remaining)]
        execute(cmd, work / 'logs' / ('train-' + args.plan + '.log'))
        if not (run_dir / 'completed.json').exists():
            print('Training saved a checkpoint. Rerun same command after restoring full output.', flush=True)
            return
        for dataset in ('sc/sst2', 'multifaced/tweeteval'):
            directory = work / 'evaluation' / ('after-' + args.plan) / dataset.replace('/', '_')
            directory.mkdir(parents=True, exist_ok=True)
            once(work, 'eval-after-' + args.plan + '-' + dataset.replace('/', '_'),
                 eval_command(args.model, manifest['model_revision'], run_dir / 'icldist/adapter', dataset, directory, args),
                 [directory / 'result.txt'])
    except Exception as exc:
        write_json(work / f'failure-{args.plan}.json', {'type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc(), 'wall_seconds': time.monotonic() - begin,
                   'note': 'No silent plan substitution; B/R datasets remain separate.'})
        raise
    finally:
        collect_report(work, args, time.monotonic() - begin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='/kaggle/working/sentiment-v1-pilot')
    parser.add_argument('--plan', choices=['A', 'B', 'B0', 'D', 'F', 'BCTD', 'R'], default='B')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--model-revision', default='main')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--local-corpus')
    parser.add_argument('--local-corpus-revision')
    parser.add_argument('--mapping-confirmed', action='store_true')
    parser.add_argument('--repair-preparation', action='store_true',
                        help='After a source-only fix, archive preparation/receipts before model work; reuse downloaded raw corpus')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--eval-samples', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--score-seconds', type=int, default=3600)
    parser.add_argument('--max-seconds', type=int, default=10800)
    parser.add_argument('--include-paper-reference', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    pipeline(parser.parse_args())


if __name__ == '__main__':
    main()
