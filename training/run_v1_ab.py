"""Run interchangeable v1 plans with immutable config and no test-driven switch.

Plan directories remain separate. Resuming a partial stage restores full state;
starting ICLDIST loads the completed KNOWDIST adapter with a new optimizer.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from distill_core import digest, file_hash, read_records, write_json
from v1_core import steps_for_tokens


def build_request(prepared, config, plan, seed, pilot_steps=0):
    prepared = Path(prepared)
    stages, shared_model, model_revision, tokenizer_hashes, pool_sha = {}, None, None, None, None
    for stage in ('knowdist', 'icldist'):
        directory = prepared / stage
        manifest = json.loads((directory / 'manifest.json').read_text())
        for split, sha in manifest['split_hashes'].items():
            if file_hash(directory / f'{split}.jsonl') != sha:
                raise ValueError(f'Prepared split changed: {stage}/{split}')
        if shared_model is not None and (manifest['model'], manifest['model_revision']) != (shared_model, model_revision):
            raise ValueError('Stages use different base model/revision')
        if tokenizer_hashes is not None and manifest['tokenizer_hashes'] != tokenizer_hashes:
            raise ValueError('Stages use different tokenizers')
        if pool_sha is not None and manifest['pool_sha256'] != pool_sha:
            raise ValueError('Stages use different candidate pools')
        if stage == 'knowdist':
            expected_method = config['plans'][plan]['selection']
        elif config['plans'][plan].get('stratified_icl'):
            expected_method = 'stfa_stratified'
        else:
            expected_method = 'coverage'
        if manifest['selection']['method'] != expected_method:
            raise ValueError(f'Stage {stage} selection does not match plan: {manifest["selection"]["method"]} != {expected_method}')
        tokenizer_hashes, pool_sha = manifest['tokenizer_hashes'], manifest['pool_sha256']
        shared_model, model_revision = manifest['model'], manifest['model_revision']
        train = [r for _, r in read_records(directory / 'train.jsonl')]
        steps, actual = steps_for_tokens(train, config['train_tokens'][stage],
                                         config['batch_size'], config['grad_accum'], seed)
        if pilot_steps:
            steps, actual = pilot_steps, None
        stages[stage] = {'prepared': str(directory.resolve()), 'manifest_sha256': file_hash(directory / 'manifest.json'),
                         'steps': steps, 'expected_train_tokens': actual,
                         'target_train_tokens': config['train_tokens'][stage]}
    source = {name: file_hash(Path(__file__).with_name(name)) for name in
              ('run_v1_ab.py', 'train_distill.py', 'distill_core.py', 'v1_core.py', 'prepare_v1_pruned.py')}
    return {'config': config, 'plan': plan, 'seed': seed, 'pilot_steps': pilot_steps,
            'source_sha256': digest(source),
            'model': shared_model, 'model_revision': model_revision, 'stages': stages}


def stage_artifacts(directory):
    files = [directory / 'summary.json', directory / 'identity.json']
    files += sorted(p for p in (directory / 'adapter').iterdir() if p.is_file())
    if not any(p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError('Completed stage has no adapter weights')
    return {str(p.relative_to(directory)): file_hash(p) for p in files}


def commands(request, output):
    config, stages = request['config'], request['stages']
    plan_cfg = config['plans'][request['plan']]
    fork_s1 = request.get('fork_stage1')
    result = []
    for stage in ('knowdist', 'icldist'):
        if stage == 'knowdist' and fork_s1:
            continue
        args = [sys.executable, str(Path(__file__).with_name('train_distill.py')), 'train',
            '--prepared', stages[stage]['prepared'], '--output', str(Path(output) / stage),
            '--steps', str(stages[stage]['steps']), '--seed', str(request['seed']),
            '--mode', plan_cfg['mode'], '--batch_size', str(config['batch_size']),
            '--grad_accum', str(config['grad_accum']), '--learning_rate', str(config['learning_rate']),
            '--rank', str(config['rank']), '--target_modules', config['target_modules'],
            '--probe_every', str(config['probe_every']), '--save_every', str(config['save_every'])]
        if plan_cfg.get('sample_normalized_loss'):
            args += ['--sample-normalized-loss']
        if plan_cfg.get('replay_decay'):
            args += ['--replay-decay', str(plan_cfg['replay_decay'])]
        if plan_cfg.get('neftune_alpha'):
            args += ['--neftune-alpha', str(plan_cfg['neftune_alpha'])]
        if plan_cfg.get('max_grad_norm'):
            args += ['--max_grad_norm', str(plan_cfg['max_grad_norm'])]
        if stage == 'icldist':
            adapter_path = fork_s1 if fork_s1 else str(Path(output) / 'knowdist' / 'adapter')
            args += ['--init_adapter', adapter_path]
        result.append((stage, args))
    return result


def run(args):
    config = json.loads(Path(args.config).read_text())
    if args.plan not in config['plans']:
        raise ValueError('Plan not configured')
    selection = json.loads((Path(args.prepared) / 'selection.json').read_text())
    if selection['method'] != config['plans'][args.plan]['selection']:
        raise ValueError('Selected dataset does not match plan; explicit switch requires another prepared directory')
    request = build_request(args.prepared, config, args.plan, args.seed, args.pilot_steps)
    if getattr(args, 'fork_stage1', None):
        request['fork_stage1'] = str(Path(args.fork_stage1).resolve())
    output = Path(args.output).resolve()
    if args.dry_run:
        print(json.dumps({'request': request, 'commands': commands(request, output)}, indent=2))
        return
    if not 0 < args.max_seconds <= 36000:
        raise ValueError('Set a per-invocation time budget in (0,36000] seconds, below 12h session limit')
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / 'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Run configuration differs; use a new output path')
    elif any(output.iterdir()):
        raise ValueError('Nonempty output without request identity')
    else:
        write_json(request_path, request)
    begin = time.monotonic()
    for stage, command in commands(request, output):
        directory = output / stage
        summary_path = directory / 'summary.json'
        if summary_path.exists() and json.loads(summary_path.read_text()).get('completed'):
            identity = json.loads((directory / 'identity.json').read_text())
            if identity['manifest_sha256'] != request['stages'][stage]['manifest_sha256']:
                raise ValueError('Completed stage manifest identity differs')
            integrity = directory / 'artifact-integrity.json'
            if not integrity.exists() or json.loads(integrity.read_text()) != stage_artifacts(directory):
                raise ValueError('Completed stage artifact integrity missing or changed; preserve and inspect')
            print(f'Reusing completed {stage}')
            continue
        if stage == 'icldist' and not (output / 'knowdist' / 'adapter' / 'adapter_config.json').is_file():
            raise ValueError('KNOWDIST is not complete; cannot start ICLDIST')
        remaining = args.max_seconds - (time.monotonic() - begin) - 120
        if remaining <= 0:
            print('Session budget exhausted; rerun this command after restoring output in next session.')
            return
        command += ['--max_seconds', str(remaining)]
        latest = directory / 'latest.json'
        if latest.exists():
            # Resolve under current writable directory, not an old /kaggle/input mount.
            name = Path(json.loads(latest.read_text())['checkpoint']).name
            command += ['--resume', str(directory / name)]
            # Resume retains init_adapter in identity, but train requires it as ancestry,
            # not as a second initialization. train handles that distinction.
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError(f'Incomplete stage has no full checkpoint: {directory}; preserve and inspect')
        subprocess.run(command, check=True)
        summary = json.loads(summary_path.read_text())
        if not summary.get('completed'):
            print(f'{stage} saved a checkpoint; resume same plan/output before continuing.')
            return
        actual = summary['state']['train_tokens']
        expected = request['stages'][stage]['expected_train_tokens']
        if expected is not None and actual != expected:
            raise ValueError(f'Training token count differs from planned batch stream: {actual} != {expected}')
        write_json(directory / 'artifact-integrity.json', stage_artifacts(directory))
    write_json(output / 'completed.json', {'request_sha256': file_hash(request_path),
               'plan': args.plan, 'seed': args.seed,
               'stage_summaries': {s: json.loads((output / s / 'summary.json').read_text())
                                   for s in ('knowdist', 'icldist')},
               'note': 'Teacher-response CE only; use an audited benchmark evaluator for F1/Accuracy.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default=str(Path(__file__).parent.parent / 'configs/v1_ab.json'))
    p.add_argument('--plan', choices=['A', 'B', 'B0', 'D', 'F', 'BCTD', 'R'], required=True)
    p.add_argument('--fork-stage1', help='Path to completed Stage 1 adapter to skip Stage 1 training')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-seconds', type=float, default=28800)
    p.add_argument('--pilot-steps', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.pilot_steps < 0:
        p.error('pilot-steps must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
