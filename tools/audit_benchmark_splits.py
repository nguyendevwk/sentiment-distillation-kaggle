"""CPU-only audit of released test integrity and exact train/val overlap.

Reads text fields only for overlap decisions, never labels. Does not alter data.
This is not a near-duplicate or distillation-corpus audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, list) or any(not isinstance(r, dict) for r in value):
        raise ValueError(f'Expected array of records: {path}')
    return value


def text_key(record):
    for field in ('sentence', 'text', 'review'):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return ' '.join(value.split()).casefold()
    raise ValueError('No supported text field; do not substitute a label or empty key')


def audit(data_root, manifest, seed=42, pilot_samples=64):
    if pilot_samples < 1:
        raise ValueError('pilot_samples must be positive')
    root = Path(data_root)
    report = {'upstream': manifest['repository'], 'revision': manifest['revision'],
              'scope': 'Exact normalized text overlap within benchmark splits only; no corpus/near-duplicate certification.',
              'seed': seed, 'pilot_samples_requested': pilot_samples, 'datasets': {}}
    all_test = set()
    for key, expected in manifest['files'].items():
        if len(Path(key).parts) != 2 or '..' in Path(key).parts or Path(key).is_absolute():
            raise ValueError('Invalid dataset key')
        path = root / key / 'test.json'
        data = rows(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        report['datasets'][key] = {'test_records': len(data), 'test_sha256': sha,
            'matches_official_release': len(data) == expected['records'] and sha == expected['sha256']}
        all_test.update(text_key(r) for r in data)
    for key, result in report['datasets'].items():
        for split in ('train', 'val'):
            path = root / key / f'{split}.json'
            if not path.exists():
                result[split] = None
                continue
            data = rows(path)
            indices = [i for i, r in enumerate(data) if text_key(r) in all_test]
            result[split] = {'records': len(data), 'exact_test_overlap_rows': len(indices), 'overlap_indices': indices}
            if split == 'val':
                sampled = random.Random(seed).sample(range(len(data)), min(pilot_samples, len(data)))
                result['pilot_val'] = {'sampled_records': len(sampled),
                    'exact_test_overlap_rows': sum(i in set(indices) for i in sampled)}
    report['test_integrity_passed'] = all(r['matches_official_release'] for r in report['datasets'].values())
    report['development_overlap_rows'] = sum(r[s]['exact_test_overlap_rows'] for r in report['datasets'].values()
                                              for s in ('train', 'val') if r[s] is not None)
    report['exact_benchmark_splits_disjoint'] = report['development_overlap_rows'] == 0
    return report


def validate_final_test_request(max_samples, data_prop):
    if max_samples != 0 or data_prop != 1:
        raise ValueError('Final test must use max_samples=0 and data_prop=1; use validation for a subset')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default=str(ROOT / 'datasets'))
    p.add_argument('--manifest', default=str(ROOT / 'configs/sentibench_v1_test_manifest.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pilot-samples', type=int, default=64)
    p.add_argument('--fail-on-overlap', action='store_true')
    p.add_argument('--check-final-test-request', action='store_true')
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--data-prop', type=float, default=1.)
    args = p.parse_args()
    if args.check_final_test_request:
        validate_final_test_request(args.max_samples, args.data_prop)
    report = audit(args.data_root, json.loads(Path(args.manifest).read_text()), args.seed, args.pilot_samples)
    path = Path(args.output)
    if path.exists():
        raise ValueError('Audit output exists; choose a new path to preserve previous evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for key, item in report['datasets'].items():
        overlap = {s: item[s]['exact_test_overlap_rows'] if item[s] else None for s in ('train', 'val')}
        print(key, 'official_test=', item['matches_official_release'], 'records=', item['test_records'], 'overlap=', overlap)
    print('Audit saved:', path)
    if not report['test_integrity_passed'] or (args.fail_on_overlap and not report['exact_benchmark_splits_disjoint']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
