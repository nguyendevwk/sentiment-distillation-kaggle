"""Evaluate existing base/adapter/paper on all 12 complete SentiBench test sets.

Run on Kaggle after cloning. No retraining; separate resumable output and caches.
Exact text audit is not a guarantee about paraphrases or base-model pretraining.
"""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
from evaluation_core import inspect_trained_adapter, locked_test_keys, prepare_clean_benchmark, sha, verify_result
from distill_core import write_json
from run_kaggle_pilot import execute


def result_paths(output, target, dataset, seed):
    directory = Path(output) / 'results' / target / dataset.replace('/', '_') / f'seed-{seed}'
    return directory, directory / 'result.txt', directory / 'done.json'


def collect(output, targets, lock, seeds):
    report = {'protocol': 'sentibench-v1-full12-corrected-fsa-r1', 'split': 'test',
              'datasets': list(lock['files']), 'seeds': seeds, 'models': {},
              'limitations': ['Same official test data; corrected FSA literal parser and cleaned demos differ from unpatched author evaluation.',
                             'One seed is preliminary; paper uses a mean over three evaluation seeds.',
                             'Exact own-training/demo checks do not certify absence of paraphrases or base/paper pretraining overlap.',
                             'Previous pilot exposed one SST2 validation input duplicated in test; do not describe test as never observed.']}
    csv_rows = []
    for target in targets:
        datasets = {}
        for dataset in lock['files']:
            records = []
            for seed in seeds:
                directory, result, receipt = result_paths(output, target, dataset, seed)
                if not receipt.exists():
                    continue
                saved = json.loads(receipt.read_text())
                if sha(result) != saved['result_sha256']:
                    raise ValueError('Completed result changed')
                lines = [json.loads(line) for line in result.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                metrics = verify_result(lines[-1], dataset, lock, seed)
                records.append(metrics)
                timing = metrics['metadata'].get('inference_measurement', {})
                csv_rows.append({'model': target, 'dataset': dataset, 'seed': seed,
                    'metric': 'sentiment_graph_f1' if dataset == 'absa/opener' else 'extraction_f1' if dataset.startswith('absa/') else 'macro_f1',
                    'f1_percent': metrics['f1'] * 100, 'accuracy_percent': metrics['acc'] * 100 if metrics['acc'] is not None else None,
                    'n': metrics['metadata']['num_samples'],
                    'inference_ms_per_example': timing.get('amortized_ms_per_example'),
                    'peak_allocated_gib': timing.get('peak_allocated_bytes', 0) / 1024**3})
            if records:
                values = [r['f1'] * 100 for r in records]
                datasets[dataset] = {'seeds_completed': len(records), 'seeds_expected': len(seeds),
                                     'mean_f1_percent': statistics.mean(values),
                                     'std_f1_pp': statistics.stdev(values) if len(values) > 1 else None}
        complete = len(datasets) == len(lock['files']) and all(d['seeds_completed'] == len(seeds) for d in datasets.values())
        report['models'][target] = {'datasets': datasets, 'complete': complete,
                                    'avg12_f1_percent': statistics.mean(d['mean_f1_percent'] for d in datasets.values()) if complete else None}
    # Cross-model equality is checked on actual rendered prompts/data and demo IDs.
    for dataset in lock['files']:
        for seed in seeds:
            fingerprints = []
            for target in targets:
                _, result, receipt = result_paths(output, target, dataset, seed)
                if receipt.exists():
                    record = json.loads([l for l in result.read_text(encoding='utf-8-sig').splitlines() if l.strip()][-1])[1]
                    meta = record['metadata']
                    fingerprints.append((meta['data_sha256'], meta['demo_manifest_sha256']))
            if len(set(fingerprints)) > 1:
                raise ValueError(f'Models used different rendered data/demo IDs for {dataset}/seed{seed}')
    write_json(Path(output) / 'full12-report.json', report)
    if csv_rows:
        with (Path(output) / 'full12-results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    names = list(targets)
    lines = ['# SentiBench full test 12 datasets', '',
             'F1 phần trăm. N/A là chưa hoàn tất, không được tính thành 0. Không chọn lại phương pháp theo test.', '',
             '| Dataset | N | ' + ' | '.join(names) + ' |',
             '|---|---:|' + '|'.join(['---:'] * len(names)) + '|']
    for dataset, spec in lock['files'].items():
        values = []
        for target in names:
            item = report['models'][target]['datasets'].get(dataset)
            if not item:
                values.append('N/A')
            else:
                std = f' ± {item["std_f1_pp"]:.2f}' if item['std_f1_pp'] is not None else ''
                values.append(f'{item["mean_f1_percent"]:.2f}{std} ({item["seeds_completed"]}/{len(seeds)} seeds)')
        lines.append('| ' + dataset + ' | ' + str(spec['records']) + ' | ' + ' | '.join(values) + ' |')
    lines.append('| Avg-12 | — | ' + ' | '.join(f'{report["models"][t]["avg12_f1_percent"]:.2f}'
                        if report['models'][t]['complete'] else 'N/A' for t in names) + ' |')
    lines += ['', 'Chi tiết Accuracy, độ trễ/VRAM, raw predictions, dữ liệu audit và protocol ở CSV/JSON/các thư mục result.',
              'FSA giữ metric gốc (Opener dùng graph F1), sửa parser/tuple order có phiên bản; không ghép với điểm cũ.']
    (Path(output) / 'full12-report.md').write_text('\n'.join(lines) + '\n')
    return report


def run(args):
    lock = json.loads((ROOT / 'configs/sentibench_v1_test_manifest.json').read_text())
    if args.dry_run:
        print(json.dumps({'datasets': {k: v['records'] for k, v in lock['files'].items()},
            'seeds': args.seeds, 'models': ['before', 'after-' + args.plan, 'paper'],
            'full_test': True, 'max_samples': 0, 'demo_split': args.demo_split,
            'work': args.pilot_work, 'output': args.output, 'model_operations': 'none in dry-run'}, indent=2))
        return
    if not Path('/kaggle/working').is_dir():
        raise ValueError('Run this on Kaggle; local model operations disabled')
    if not 0 < args.max_seconds <= 36000:
        raise ValueError('max-seconds must be in (0,36000]')
    output, work = Path(args.output).resolve(), Path(args.pilot_work).resolve()
    if output == work or not output.is_relative_to(Path('/kaggle/working').resolve()):
        raise ValueError('Use a separate writable evaluation output under /kaggle/working')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'logs').mkdir(exist_ok=True)
    os.environ['PYTHONHASHSEED'] = '0'  # Stable set iteration in inherited author code, in child evaluators.
    run_dir = work / 'runs' / f'pilot-{args.pilot_steps}' / args.plan / f'seed-{args.training_seed}'
    source = {str(p.relative_to(ROOT)): sha(p) for p in
              [Path(__file__), ROOT / 'evaluation_core.py', ROOT / 'notebook_runtime.py',
               ROOT / 'icl_multifaced_kaggle.py', ROOT / 'fsa_kaggle.py', ROOT / 'utils.py',
               ROOT / 'icl_multifaced.py', ROOT / 'fsa.py',
               ROOT / 'tools/audit_benchmark_splits.py', ROOT / 'tools/run_kaggle_pilot.py',
               *sorted((ROOT / 'parse_utils').glob('*.py'))]}
    request = {'source': source, 'pilot_work': str(work), 'plan': args.plan, 'run_dir': str(run_dir),
               'test_snapshot': lock, 'seeds': args.seeds, 'demo_split': args.demo_split,
               'classification_batch_size': args.classification_batch_size, 'fsa_batch_size': args.fsa_batch_size,
               'max_seq_length': args.max_length, 'max_new_tokens': args.max_new_tokens,
               'python_hash_seed': 0}
    request_file = output / 'evaluation-request.json'
    if request_file.exists() and json.loads(request_file.read_text()) != request:
        raise ValueError('Evaluation config/source changed; keep previous results and use a new output')
    write_json(request_file, request)
    started = time.monotonic()
    targets = {'before': {}, 'after-' + args.plan: {}, 'paper': {}}
    try:
        from run_kaggle_pilot import environment
        env = environment(output)
        env_lock = {'packages': env['packages'], 'python': env['python'],
                    'gpus': env['gpus'], 'cuda_version': env['cuda_version'],
                    'nltk': version('nltk')}
        env_file = output / 'environment-lock.json'
        if env_file.exists() and json.loads(env_file.read_text()) != env_lock:
            raise ValueError('GPU/software changed; restore the original environment or use a new output to avoid mixing timings')
        write_json(env_file, env_lock)
        tests = locked_test_keys(ROOT / 'datasets', lock)
        cleaned = output / 'benchmark'
        cleaning = prepare_clean_benchmark(ROOT / 'datasets', cleaned, lock)
        own_audit = inspect_trained_adapter(work, run_dir, tests)
        write_json(output / 'own-training-exact-audit.json', own_audit)
        if own_audit['exact_checked_overlap_found']:
            raise ValueError('Consumed training data contains test text; audit saved. Evaluation cannot undo training leakage. Keep this checkpoint as historical and retrain cleanly.')
        base_model, base_revision = own_audit['base_model'], own_audit['base_revision']
        from huggingface_hub import HfApi
        paper_file = output / 'paper-model.json'
        if not paper_file.exists():
            model = 'zhang-yice/Qwen2.5-1.5B-sentiment-distillation-v1'
            write_json(paper_file, {'model': model, 'revision': HfApi().model_info(model).sha,
                                   'training_corpus_audit': 'not available in full; external reference'})
        paper = json.loads(paper_file.read_text())
        targets = {'before': {'model': base_model, 'revision': base_revision, 'adapter': None},
                   'after-' + args.plan: {'model': base_model, 'revision': base_revision, 'adapter': str(run_dir / 'icldist/adapter')},
                   'paper': {**paper, 'adapter': None}}
        write_json(output / 'targets.json', targets)
        # Dataset-major order compares all three models early; deterministic jobs.
        for dataset in lock['files']:
            if not (cleaned / dataset / f'{args.demo_split}.json').exists():
                raise ValueError(f'No {args.demo_split} demo file for {dataset}; no silent split fallback')
            for seed in args.seeds:
                for name, target in targets.items():
                    directory, result, receipt = result_paths(output, name, dataset, seed)
                    if receipt.exists():
                        if sha(result) != json.loads(receipt.read_text())['result_sha256']:
                            raise ValueError('Stored result changed')
                        continue
                    remaining = args.max_seconds - (time.monotonic() - started) - 180
                    if remaining <= 0:
                        print('Session budget reached. Restore full evaluation output and rerun the same command.', flush=True)
                        return
                    directory.mkdir(parents=True, exist_ok=True)
                    fsa = dataset.startswith('absa/')
                    cmd = [sys.executable, str(ROOT / ('fsa_kaggle.py' if fsa else 'icl_multifaced_kaggle.py')),
                           '--dataset', dataset, '--subname', 'full12_' + name, '--seed', str(seed),
                           '--model_name_or_path', target['model'], '--model_revision', target['revision'],
                           '--data_dir', str(cleaned), '--output_dir', str(directory), '--eval_split', 'test',
                           '--demo_split', args.demo_split, '--max_samples', '0', '--data_prop', '1',
                           '--eval_batch_size', str(args.fsa_batch_size if fsa else args.classification_batch_size),
                           '--max_seq_length', str(args.max_length), '--max_new_tokens', str(args.max_new_tokens),
                           '--k_shot', '4', '--measure_inference', '--resume_file', str(directory / 'predictions.checkpoint.json'),
                           '--max_eval_seconds', str(remaining)]
                    if target['adapter']:
                        cmd += ['--adapter', target['adapter']]
                    execute(cmd, output / 'logs' / f'{dataset.replace("/", "_")}-{seed}-{name}.log')
                    if not result.exists():
                        print('Partial predictions saved. Rerun same command to resume.', flush=True)
                        return
                    record = json.loads([l for l in result.read_text(encoding='utf-8-sig').splitlines() if l.strip()][-1])
                    metrics = verify_result(record, dataset, lock, seed)
                    if metrics['metadata']['inference_measurement']['measured_examples'] != lock['files'][dataset]['records']:
                        raise ValueError('Timing/prediction cache coverage is incomplete')
                    write_json(receipt, {'result_sha256': sha(result), 'test_sha256': lock['files'][dataset]['sha256']})
                    collect(output, targets, lock, args.seeds)
    except Exception as exc:
        write_json(output / f'failure-{time.time_ns()}.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        collect(output, targets, lock, args.seeds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-work', default='/kaggle/working/v1-pilot-01')
    p.add_argument('--output', default='/kaggle/working/full12-evaluation-01')
    p.add_argument('--plan', choices=['A', 'B', 'R'], default='B')
    p.add_argument('--pilot-steps', type=int, default=50)
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--demo-split', choices=['train', 'val'], default='train',
                   help='Explicit protocol: train supports all12; val is missing for ATSA/ACSA and will fail, never fall back')
    p.add_argument('--classification-batch-size', type=int, default=1)
    p.add_argument('--fsa-batch-size', type=int, default=1)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=150)
    p.add_argument('--max-seconds', type=float, default=28800)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.classification_batch_size, args.fsa_batch_size, args.max_new_tokens) < 1:
        p.error('Seeds must be unique; batches and generation length positive')
    run(args)


if __name__ == '__main__':
    main()
