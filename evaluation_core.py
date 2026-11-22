"""CPU-only full-test data locking and exact leakage checks. No model imports."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import time

from tools.audit_benchmark_splits import audit, rows, text_key


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def locked_test_keys(data_root, lock):
    result = audit(data_root, lock)
    if not result['test_integrity_passed']:
        raise ValueError('Test files differ from official released SentiBench snapshot')
    return {text_key(r) for key in lock['files'] for r in rows(Path(data_root) / key / 'test.json')}


def prepare_clean_benchmark(data_root, output, lock):
    """Filter train/val only against ALL official test texts; copy test byte-for-byte."""
    root, target = Path(data_root), Path(output)
    tests = locked_test_keys(root, lock)
    request = {f'{key}/{split}': sha(root / key / f'{split}.json')
               for key in lock['files'] for split in ('test', 'train', 'val')
               if (root / key / f'{split}.json').is_file()}
    manifest_path = target / 'cleaning.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['input_hashes'] != request or manifest['source_sha256'] != sha(__file__):
            raise ValueError('Cleaning source or inputs changed; choose a new evaluation output')
        for name, expected in manifest['output_hashes'].items():
            if sha(target / name) != expected:
                raise ValueError('Cleaned benchmark changed')
        return manifest
    if target.exists() and any(target.iterdir()):
        raise ValueError('Incomplete cleaned benchmark output; preserve it and choose a new output')
    staging = target.with_name(f'.{target.name}.pending-{time.time_ns()}')
    staging.mkdir(parents=True)
    removed = {}
    for key in lock['files']:
        destination = staging / key
        destination.mkdir(parents=True)
        shutil.copy2(root / key / 'test.json', destination / 'test.json')
        for split in ('train', 'val'):
            path = root / key / f'{split}.json'
            if not path.exists():
                continue
            kept, rejected = [], []
            for index, r in enumerate(rows(path)):
                if text_key(r) in tests:
                    rejected.append(index)
                    continue
                r['_demo_id'] = stable_digest([key, split, index, text_key(r)])
                kept.append(r)
            (destination / f'{split}.json').write_text(json.dumps(kept, ensure_ascii=False))
            removed[f'{key}/{split}'] = {'removed_indices': rejected, 'removed': len(rejected), 'retained': len(kept)}
    check = audit(staging, lock)
    if not check['test_integrity_passed'] or not check['exact_benchmark_splits_disjoint']:
        raise ValueError(f'Clean benchmark failed exact audit; retained diagnostic files at {staging}')
    manifest = {'input_hashes': request, 'source_sha256': sha(__file__), 'removed': removed,
        'test_snapshot': lock, 'audit': check,
        'output_hashes': {str(p.relative_to(staging)): sha(p) for p in staging.glob('*/*/*.json')},
        'scope': 'Exact whitespace/casefold text disjointness; no near-duplicate or pretraining certification.'}
    (staging / 'cleaning.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    staging.rename(target)
    return manifest


def embedded_input_keys(prompt):
    """Audit complete marked input blocks, including demonstrations, without labels."""
    starts = list(re.finditer(r'(?m)^(?:Sentence|Sentence1|Premise|Input|Text|Review|Tweet|News Headline):[ \t]*', prompt))
    result = []
    for i, match in enumerate(starts):
        ending = re.search(r'(?m)^(?:Output|Label|Answer):', prompt[match.end():])
        end = match.end() + ending.start() if ending else len(prompt)
        if i + 1 < len(starts):
            end = min(end, starts[i + 1].start())
        text = ' '.join(prompt[match.end():end].split()).casefold()
        if text:
            result.append(text)
    return result


def inspect_trained_adapter(pilot_work, run_dir, tests):
    """Audit actual presented records (including skipped-update batches), not unused pool.

    Uses saved config/manifest and exact trainer shuffle, never pickle state.
    Exact-source and recognized marked-demo checks cannot certify paraphrase absence.
    """
    work, run = Path(pilot_work), Path(run_dir)
    completed = json.loads((run / 'completed.json').read_text())
    if completed['request_sha256'] != sha(run / 'request.json'):
        raise ValueError('Completed training request changed')
    request = json.loads((run / 'request.json').read_text())
    report = {'stages': {}, 'confirmed_overlap_records': [],
              'scope': 'Actual consumed source keys + complete supported marked inputs; no near-duplicate/proxy/pretraining certification.'}
    test_hashes = {stable_digest(text) for text in tests}
    model = revision = None
    for stage in ('knowdist', 'icldist'):
        directory = run / stage
        integrity = json.loads((directory / 'artifact-integrity.json').read_text())
        for name, expected in integrity.items():
            if sha(directory / name) != expected:
                raise ValueError(f'Training artifact changed: {stage}/{name}')
        summary = json.loads((directory / 'summary.json').read_text())
        if not summary['completed']:
            raise ValueError('Train stage is incomplete')
        identity = json.loads((directory / 'identity.json').read_text())
        config = identity['config']
        prepared = work / 'prepared' / request['plan'] / stage
        if sha(prepared / 'manifest.json') != identity['manifest_sha256']:
            raise ValueError('Training preparation manifest differs from identity')
        manifest = json.loads((prepared / 'manifest.json').read_text())
        if sha(prepared / 'train.jsonl') != manifest['split_hashes']['train']:
            raise ValueError('Training rows changed')
        if model is not None and (model, revision) != (manifest['model'], manifest['model_revision']):
            raise ValueError('Training stages use different base revisions')
        model, revision = manifest['model'], manifest['model_revision']
        candidates = [json.loads(line) for line in (prepared / 'train.jsonl').read_text().splitlines() if line.strip()]
        if not candidates:
            raise ValueError('Training rows are empty')
        n = summary['state']['step'] * config['batch_size'] * config['grad_accum']
        consumed, epoch = [], 0
        while len(consumed) < n:
            order = list(range(len(candidates)))
            random.Random(config['seed'] + epoch).shuffle(order)
            consumed.extend(candidates[i] for i in order[:n - len(consumed)])
            epoch += 1
        if sum(len(r['input_ids']) for r in consumed) != summary['state']['train_tokens']:
            raise ValueError('Could not reconstruct actual consumed training stream')
        for r in {r['id']: r for r in consumed}.values():
            reason = []
            if r['source_key'] in test_hashes:
                reason.append('final_source_input')
            if set(embedded_input_keys(r['prompt'])) & tests:
                reason.append('marked_prompt_or_demo_input')
            if reason:
                report['confirmed_overlap_records'].append({'stage': stage, 'id': r['id'], 'reason': reason})
        report['stages'][stage] = {'presentations': n, 'unique_ids': len({r['id'] for r in consumed}),
                                  'train_tokens': summary['state']['train_tokens'], 'updates': summary['state']['optimizer_updates']}
    report.update(base_model=model, base_revision=revision,
                  exact_checked_overlap_found=bool(report['confirmed_overlap_records']),
                  adapter_hashes={p.name: sha(p) for p in sorted((run / 'icldist/adapter').iterdir()) if p.is_file()})
    return report


def verify_result(record, dataset, lock, seed):
    metadata = record[1]['metadata']
    settings = metadata['settings']
    if metadata['dataset'] != dataset or metadata['seed'] != seed:
        raise ValueError('Result dataset/seed mismatch')
    if settings['eval_split'] != 'test' or settings['max_samples'] != 0 or settings['data_prop'] != 1:
        raise ValueError('A partial/development result cannot count as final full test')
    if metadata['num_samples'] != lock['files'][dataset]['records']:
        raise ValueError('Result does not contain the complete official test set')
    return record[1]
