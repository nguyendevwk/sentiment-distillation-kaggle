"""Kaggle v1 A/B preparation. No heavy imports until tokenize/score/download.

Plan A: coverage + base-student conditional/unconditional response-loss ratio.
Plan B: coverage only, available without model-scoring cost.
Both plans share random control, ICLDIST IDs and development splits.
"""
import argparse
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import time
from functools import wraps

from distill_core import digest, file_hash, read_records, write_json
from v1_core import (canonical_record, iter_objects, limited_candidates, normalized_text,
                     select_rows, select_rows_stfa, split_name)


def parse_token_budget(val):
    if not val:
        return {'knowdist': 1500000, 'icldist': 500000}
    v = str(val).lower().strip()
    if v == '2m':
        return {'knowdist': 1500000, 'icldist': 500000}
    elif v == '4m':
        return {'knowdist': 3000000, 'icldist': 1000000}
    elif v.endswith('m'):
        total = int(float(v[:-1]) * 1000000)
        return {'knowdist': int(total * 0.75), 'icldist': int(total * 0.25)}
    elif v.isdigit():
        total = int(v)
        return {'knowdist': int(total * 0.75), 'icldist': int(total * 0.25)}
    return {'knowdist': 1500000, 'icldist': 500000}


def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output is not empty: {path}; choose a new experiment directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_preparation(func):
    """Publish a preparation directory only after the entire step succeeds.

    Failed attempts are kept beside the destination for diagnosis, never deleted.
    The original Namespace is unchanged so callers retain the final output path.
    """
    @wraps(func)
    def wrapped(args):
        target = Path(args.output)
        if target.exists() and any(target.iterdir()):
            raise ValueError(f'Output is not empty: {target}; use launcher recovery or a new output')
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f'.{target.name}.pending-{time.time_ns()}')
        copied = argparse.Namespace(**vars(args))
        copied.output = str(staging)
        try:
            result = func(copied)
            staging.rename(target)
            return result
        except BaseException:
            if staging.exists():
                failed = staging.with_name(staging.name.replace('.pending-', '.failed-', 1))
                staging.rename(failed)
                print(f'Incomplete preparation preserved: {failed}', flush=True)
            raise
    return wrapped


def jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().dataset_info('zhang-yice/sentiment-distillation-v1', revision=args.revision).sha
    directory = snapshot_download('zhang-yice/sentiment-distillation-v1', repo_type='dataset',
                                  revision=revision, local_dir=args.output,
                                  allow_patterns=['KNOWDIST_200k/**', 'ICLDIST_100k/**'])
    write_json(Path(args.output) / 'download-manifest.json', {'dataset': 'zhang-yice/sentiment-distillation-v1',
                                                            'revision': revision, 'directory': directory})
    print(f'Downloaded v1 revision {revision}; inspect schemas before normalize.')


@atomic_preparation
def normalize(args):
    if not args.mapping_confirmed:
        raise ValueError('Inspect full source records/extraction rules; then pass --mapping-confirmed')
    root, out = Path(args.input), fresh(args.output)
    benchmark_keys, benchmark_files = set(), {}
    if args.benchmark_root:
        files = sorted(Path(args.benchmark_root).glob('*/*/test.json'))
        if not files:
            raise ValueError('No benchmark */*/test.json files found')
        for path in files:
            benchmark_files[str(path.relative_to(args.benchmark_root))] = file_hash(path)
            for _, obj in iter_objects(path):
                text = next((obj[k] for k in ('sentence', 'text', 'review')
                             if isinstance(obj.get(k), str) and obj[k].strip()), None)
                if text is None:
                    raise ValueError(f'Unsupported benchmark text schema: {path}; audit required')
                benchmark_keys.add(digest(normalized_text(text)))
    files = sorted(p for p in root.rglob('*') if p.suffix in ('.json', '.jsonl')
                   and not p.name.endswith('manifest.json') and '.cache' not in p.parts)
    if not files:
        raise ValueError('No JSON/JSONL corpus files found')
    counts, seen, hashes = Counter(), set(), {}
    with (out / 'canonical.jsonl').open('w', encoding='utf-8') as stream, (out / 'exclusions.jsonl').open('w', encoding='utf-8') as exclusions:
        for path in files:
            name, sha = str(path.relative_to(root)), file_hash(path)
            hashes[name] = sha
            for index, obj in iter_objects(path):
                try:
                    r = canonical_record(obj, name, index, sha)
                    key = digest([r['stage'], r['prompt'], r['response']])
                    if key in seen:
                        raise ValueError('duplicate_prompt_response')
                    if r['source_key'] in benchmark_keys:
                        raise ValueError('exact_normalized_benchmark_test_overlap')
                    seen.add(key)
                    r['split'] = split_name(r['source_key'], args.seed, args.dev_fraction)
                    stream.write(json.dumps(r, ensure_ascii=False) + '\n')
                    counts[f'{r["stage"]}/{r["split"]}/{r["group"]}'] += 1
                except ValueError as exc:
                    exclusions.write(json.dumps({'file': name, 'position': index, 'id': f'{sha}:{index}',
                                                 'reason': str(exc)}, ensure_ascii=False) + '\n')
                    counts['excluded'] += 1
    if not any(k.startswith('knowdist/train/') for k in counts) or not any(k.startswith('icldist/train/') for k in counts):
        raise ValueError('Missing train stage; inspect exclusions and mappings')
    write_json(out / 'manifest.json', {'protocol': 'v1-ab-r1', 'corpus_revision': args.revision,
                'files': hashes, 'counts': dict(counts), 'seed': args.seed, 'dev_fraction': args.dev_fraction,
                'source_extractor': 'final_input_marker_v1',
                'source_sha256': digest([file_hash(Path(__file__)), file_hash(Path(__file__).with_name('v1_core.py'))]),
                'canonical_sha256': file_hash(out / 'canonical.jsonl'), 'benchmark_files': benchmark_files,
                'limitations': ['Exact normalized final-input overlap only; no near-duplicate/substring audit.',
                               'Embedded demonstrations are not isolated across splits; audit needed before generalization claims.',
                               'Unsupported source templates are excluded; inspect coverage and exclusion rates.']})
    print(json.dumps(dict(counts), indent=2))


@atomic_preparation
def tokenize(args):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    source, out = Path(args.normalized), fresh(args.output)
    manifest = json.loads((source / 'manifest.json').read_text())
    if file_hash(source / 'canonical.jsonl') != manifest['canonical_sha256']:
        raise ValueError('Canonical source changed')
    revision = HfApi().model_info(args.model, revision=args.model_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError('Chat template and EOS required')
    if args.max_length < 3:
        raise ValueError('max_length too small')
    candidates = limited_candidates((r for _, r in read_records(source / 'canonical.jsonl')),
            args.seed, args.per_group, {'knowdist': args.know_candidates, 'icldist': args.icl_candidates})
    retained, excluded = [], []
    for r in candidates:
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
        answer = tokenizer.encode(r['response'], add_special_tokens=False)
        if not prompt_ids or not answer:
            excluded.append({'id': r['id'], 'reason': 'empty_tokenized_prompt_or_response'})
            continue
        if answer[-1] != tokenizer.eos_token_id:
            answer.append(tokenizer.eos_token_id)
        if len(prompt_ids) + len(answer) > args.max_length:
            excluded.append({'id': r['id'], 'reason': 'whole_record_overlength'})
            continue
        if r['stage'] == 'knowdist' and getattr(args, 'rpr_filter', False):
            words = r['response'].split()
            rpr = len(r['response']) / max(1, len(r['prompt']))
            if len(words) < 5 or rpr < 0.04:
                excluded.append({'id': r['id'], 'reason': 'rpr_too_short'})
                continue
            if len(words) > 30 and (len(set(words)) / len(words)) < 0.30:
                excluded.append({'id': r['id'], 'reason': 'rpr_repetitive'})
                continue
        # Drop raw source_text from tokenized pool; final prompt/response remain unchanged.
        r = {k: v for k, v in r.items() if k != 'source_text'}
        r.update(input_ids=prompt_ids + answer, labels=[-100] * len(prompt_ids) + answer,
                 prompt_tokens=len(prompt_ids), response_tokens=len(answer), train_group=r['group'])
        retained.append(r)
    for stage in ('knowdist', 'icldist'):
        for split in ('train', 'probe', 'holdout'):
            if not any(r['stage'] == stage and r['split'] == split for r in retained):
                raise ValueError(f'Empty {stage}/{split}; expand corpus/candidate/dev limits')
    tokenizer.save_pretrained(out / 'tokenizer')
    jsonl(out / 'pool.jsonl', retained)
    jsonl(out / 'exclusions.jsonl', excluded)
    write_json(out / 'manifest.json', {'protocol': 'v1-ab-r1', 'model': args.model, 'model_revision': revision,
        'corpus_revision': manifest['corpus_revision'], 'normalization_manifest_sha256': file_hash(source / 'manifest.json'),
        'pool_sha256': file_hash(out / 'pool.jsonl'), 'seed': args.seed, 'max_length': args.max_length,
        'counts': dict(Counter(f'{r["stage"]}/{r["split"]}/{r["group"]}' for r in retained)),
        'candidate_caps': {'knowdist': args.know_candidates, 'icldist': args.icl_candidates, 'per_group': args.per_group},
        'excluded': len(excluded), 'tokenizer_template_sha256': digest(tokenizer.chat_template),
        'tokenizer_hashes': {p.name: file_hash(p) for p in sorted((out / 'tokenizer').iterdir()) if p.is_file()}})
    write_json(out / 'preview.json', [{k: r[k] for k in ('id', 'stage', 'group', 'split', 'prompt', 'response')}
                                     for r in retained[:16]])
    print(f'Retained {len(retained)} candidates; excluded {len(excluded)}. Inspect preview and coverage.')


def load_pool(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if file_hash(directory / 'pool.jsonl') != manifest['pool_sha256']:
        raise ValueError('Tokenized pool changed')
    for name, sha in manifest['tokenizer_hashes'].items():
        if file_hash(directory / 'tokenizer' / name) != sha:
            raise ValueError('Tokenizer changed')
    return manifest, [r for _, r in read_records(directory / 'pool.jsonl')]


def score(args):
    """Base-student IFD-style proxy; no claim of reproducing pre-experience IFD."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_distill import per_example_loss
    manifest, pool = load_pool(args.pool)
    if not torch.cuda.is_available():
        raise ValueError('Score on Kaggle CUDA; local model execution is not part of this workflow')
    rows = [r for r in pool if r['stage'] == 'knowdist' and r['split'] == 'train']
    if args.max_seconds < 0:
        raise ValueError('max-seconds must be nonnegative')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    identity = {'pool_sha256': manifest['pool_sha256'], 'model': manifest['model'],
        'model_revision': manifest['model_revision'], 'tokenizer_hashes': manifest['tokenizer_hashes'],
        'recipe': 'base_student_chat_response_loss_ratio_v1', 'dtype': str(dtype),
        'source_sha256': digest([file_hash(Path(__file__)), file_hash(Path(__file__).with_name('train_distill.py'))]),
        'versions': {p: version(p) for p in ('torch', 'transformers')}, 'gpu': torch.cuda.get_device_name(0)}
    out = Path(args.output)
    scores, prior_elapsed = {}, 0.
    if args.resume:
        saved = json.loads((out / 'identity.json').read_text())
        if saved != identity:
            raise ValueError('Score resume identity differs')
        for _, r in (read_records(out / 'scores.jsonl') if (out / 'scores.jsonl').exists() else []):
            if r['id'] in scores:
                raise ValueError('Duplicate score ID')
            scores[r['id']] = r
        if (out / 'summary.json').exists():
            prior_elapsed = json.loads((out / 'summary.json').read_text())['elapsed_seconds']
        prior_elapsed = max(prior_elapsed, max((r.get('elapsed_seconds', 0.) for r in scores.values()), default=0.))
    else:
        out = fresh(out)
        write_json(out / 'identity.json', identity)
    begin = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(Path(args.pool) / 'tokenizer')
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], revision=manifest['model_revision'],
               torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
    model.eval()
    completed = True
    with (out / 'scores.jsonl').open('a', encoding='utf-8') as stream, torch.inference_mode():
        for r in rows:
            if r['id'] in scores:
                continue
            ids = torch.tensor([r['input_ids']], device='cuda')
            labels = torch.tensor([r['labels']], device='cuda')
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            conditioned = float(per_example_loss(output.logits, labels)[0])
            del output, ids, labels
            answer = r['input_ids'][r['prompt_tokens']:]
            ids = torch.tensor([[bos] + answer], device='cuda')
            labels = torch.tensor([[-100] + answer], device='cuda')
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            unconditioned = float(per_example_loss(output.logits, labels)[0])
            del output, ids, labels
            if not all(math.isfinite(x) and x > 0 for x in (conditioned, unconditioned)):
                raise ValueError(f'Invalid response losses: {r["id"]}')
            record = {'id': r['id'], 'conditioned_ce': conditioned, 'unconditioned_ce': unconditioned,
                      'ifd_proxy': conditioned / unconditioned,
                      'elapsed_seconds': prior_elapsed + time.monotonic() - begin}
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            scores[r['id']] = record
            if args.max_seconds and time.monotonic() - begin >= args.max_seconds:
                completed = len(scores) == len(rows)
                break
    torch.cuda.synchronize()
    write_json(out / 'summary.json', {'completed': completed, 'records': len(scores), 'expected': len(rows),
        'elapsed_seconds': prior_elapsed + time.monotonic() - begin,
        'identity_sha256': file_hash(out / 'identity.json'), 'scores_sha256': file_hash(out / 'scores.jsonl')})
    print(f'Scored {len(scores)}/{len(rows)}. Partial scores cannot select A; B remains available.')


@atomic_preparation
def select(args):
    manifest, pool = load_pool(args.pool)
    scores = None
    score_hash = None
    if args.method == 'student':
        if not args.scores:
            raise ValueError('Plan A requires --scores; use coverage explicitly for Plan B')
        directory = Path(args.scores)
        identity = json.loads((directory / 'identity.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        if not summary['completed'] or identity['pool_sha256'] != manifest['pool_sha256']:
            raise ValueError('Incomplete scores or wrong pool; no silent fallback')
        if identity['model'] != manifest['model'] or identity['model_revision'] != manifest['model_revision']:
            raise ValueError('Scores belong to another model')
        score_hash = file_hash(directory / 'scores.jsonl')
        if score_hash != summary['scores_sha256'] or file_hash(directory / 'identity.json') != summary['identity_sha256']:
            raise ValueError('Score artifacts changed')
        scored_rows = [r for _, r in read_records(directory / 'scores.jsonl')]
        scores = {r['id']: r['ifd_proxy'] for r in scored_rows}
        expected_ids = {r['id'] for r in pool if r['stage'] == 'knowdist' and r['split'] == 'train'}
        if len(scores) != len(scored_rows) or set(scores) != expected_ids:
            raise ValueError('Score IDs do not exactly match KNOWDIST train candidates')
    if getattr(args, 'token_budget', None):
        b = parse_token_budget(args.token_budget)
        args.know_tokens = b['knowdist']
        args.icl_tokens = b['icldist']
    out = fresh(args.output)
    reports = {}
    for stage, budget, cap in [('knowdist', args.know_tokens, args.know_records),
                               ('icldist', args.icl_tokens, args.icl_records)]:
        candidates = [r for r in pool if r['stage'] == stage and r['split'] == 'train']
        if stage == 'icldist' and getattr(args, 'disable_diversification', False):
            candidates = [r for r in candidates if 'diversification' not in r.get('group', '').lower()
                          and 'diversification' not in r.get('kind', '').lower()]
        if stage == 'icldist' and getattr(args, 'stratified_icl', False):
            dev_scores = None
            if getattr(args, 'dev_scores_json', None) and Path(args.dev_scores_json).exists():
                dev_scores = json.loads(Path(args.dev_scores_json).read_text())
            train, report = select_rows_stfa(
                candidates, budget, cap, args.seed,
                dev_scores=dev_scores,
                stfa_temp=getattr(args, 'stfa_temp', 1.0),
                disable_diversification=getattr(args, 'disable_diversification', False)
            )
        else:
            method = args.method if stage == 'knowdist' else 'coverage'
            train, report = select_rows(candidates, method, budget, cap, args.seed,
                                       scores, args.min_ifd, args.max_ifd)
            if method == 'student' and report['missing_groups'] and not args.allow_missing_groups:
                raise ValueError(f'Plan A lost groups: {report["missing_groups"]}; inspect interval or run B explicitly')
        reports[stage] = report
        stage_out = out / stage
        stage_out.mkdir()
        shutil.copytree(Path(args.pool) / 'tokenizer', stage_out / 'tokenizer')
        splits = {'train': train}
        splits.update({s: [r for r in pool if r['stage'] == stage and r['split'] == s]
                       for s in ('probe', 'holdout')})
        for name, rows in splits.items():
            if not rows:
                raise ValueError(f'Empty {stage}/{name}')
            jsonl(stage_out / f'{name}.jsonl', rows)
        prepared = {k: manifest[k] for k in ('model', 'model_revision', 'corpus_revision',
                                           'tokenizer_hashes', 'tokenizer_template_sha256')}
        prepared.update(protocol='v1-ab-r1', stage=stage, selection=report,
            pool_sha256=manifest['pool_sha256'], config={'seed': args.seed, 'method': args.method,
            'score_hash': score_hash, 'ifd_interval': [args.min_ifd, args.max_ifd]},
            split_hashes={s: file_hash(stage_out / f'{s}.jsonl') for s in splits},
            counts={s: len(rows) for s, rows in splits.items()},
            train_groups=dict(Counter(r['train_group'] for r in train)),
            train_input_tokens=sum(len(r['input_ids']) for r in train),
            train_response_tokens=sum(r['response_tokens'] for r in train))
        write_json(stage_out / 'manifest.json', prepared)
        write_json(stage_out / 'selected_ids.json', [r['id'] for r in train])
    write_json(out / 'selection.json', {'method': args.method, 'seed': args.seed, 'stages': reports,
                                      'pool_sha256': manifest['pool_sha256'], 'scores_sha256': score_hash})
    print(json.dumps(reports, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('download')
    p.add_argument('--revision', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(func=download)
    p = commands.add_parser('normalize')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--benchmark-root')
    p.add_argument('--mapping-confirmed', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dev-fraction', type=float, default=.1)
    p.set_defaults(func=normalize)
    p = commands.add_parser('tokenize')
    p.add_argument('--normalized', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--know-candidates', type=int, default=10000)
    p.add_argument('--icl-candidates', type=int, default=5000)
    p.add_argument('--per-group', type=int, default=1000)
    p.add_argument('--rpr-filter', action='store_true', help='Filter knowdist noise using response-to-prompt ratio (RPR)')
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=tokenize)
    p = commands.add_parser('score')
    p.add_argument('--pool', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-seconds', type=float, default=7200)
    p.set_defaults(func=score)
    p = commands.add_parser('select')
    p.add_argument('--pool', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=['random', 'coverage', 'student'], required=True)
    p.add_argument('--scores')
    p.add_argument('--min-ifd', type=float, default=0.)
    p.add_argument('--max-ifd', type=float, default=1.)
    p.add_argument('--allow-missing-groups', action='store_true', help='Explicitly accept/report loss of group coverage in A')
    p.add_argument('--know-tokens', type=int, default=1500000)
    p.add_argument('--icl-tokens', type=int, default=500000)
    p.add_argument('--know-records', type=int, default=3500)
    p.add_argument('--icl-records', type=int, default=1500)
    p.add_argument('--token-budget', help="Preset token budget like '2m' or '4m'")
    p.add_argument('--stratified-icl', action='store_true', help="Enable Capability-Aware STFA for ICLDIST")
    p.add_argument('--stfa-temp', type=float, default=1.0, help="Temperature for STFA gap allocation")
    p.add_argument('--dev-scores-json', help="Path to JSON file with zero-shot dev scores per task (sc, multifaced, fsa)")
    p.add_argument('--disable-diversification', action='store_true', help="Ablation: exclude diversification samples from ICLDIST")
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=select)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
