"""Kaggle pilot: offline-response distillation, with uniform/adaptive group loss.

No teacher model, generated labels, or benchmark test feedback is used.
Heavy dependencies are imported only by prepare/train/export commands.
"""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

from distill_core import (GroupWeights, digest, file_hash, mapped_record, read_records,
                          split_records, write_json)


def audit(args):
    fields, groups, examples = Counter(), Counter(), []
    count = 0
    rng = random.Random(42)
    for line, obj in read_records(args.file):
        count += 1
        fields.update(obj.keys())
        groups[str(obj.get(args.group_field, '<missing>'))] += 1
        if len(examples) < args.examples:
            examples.append({'line': line, 'record': obj})
        else:
            index = rng.randrange(count)
            if index < args.examples:
                examples[index] = {'line': line, 'record': obj}
    report = {'file': str(Path(args.file).resolve()), 'sha256': file_hash(args.file),
              'records': count, 'fields': dict(fields), 'groups': dict(groups), 'examples': examples}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare(args):
    if not args.mapping_confirmed:
        raise ValueError('Inspect audit.json first; then explicitly pass --mapping_confirmed with verified field names.')
    if args.max_length < 2:
        raise ValueError('max_length must include prompt and response')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Prepared output must be empty; choose a new directory to preserve prior manifests.')
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revision = HfApi().model_info(args.model, revision=args.model_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError('Tokenizer requires chat template and EOS')
    records, rejected, seen = [], [], set()
    source_sha = file_hash(args.file)
    fields = args.prompt_fields.split(',')
    for line, obj in read_records(args.file):
        record_id = f'{source_sha}:{line}'
        try:
            r = mapped_record(obj, fields, args.response_field, args.group_field, args.source_field)
            key = digest([r['prompt'], r['response'], r['group']])
            if key in seen:
                raise ValueError('duplicate prompt/response/group')
            seen.add(key)
            prefix = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': r['prompt']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
            response_ids = tokenizer.encode(r['response'], add_special_tokens=False)
            if not response_ids:
                raise ValueError('empty response tokens')
            if response_ids[-1] != tokenizer.eos_token_id:
                response_ids.append(tokenizer.eos_token_id)
            if len(prompt_ids) + len(response_ids) > args.max_length:
                raise ValueError('overlength: whole record excluded; no response truncation')
            r.update(id=record_id, input_ids=prompt_ids + response_ids,
                     labels=[-100] * len(prompt_ids) + response_ids,
                     prompt_tokens=len(prompt_ids), response_tokens=len(response_ids))
            records.append(r)
        except ValueError as exc:
            rejected.append({'id': record_id, 'reason': str(exc)})
    train, probe, holdout, mapping = split_records(
        records, args.seed, args.dev_fraction, args.max_train, args.probe_per_group,
        args.holdout_per_group, args.min_group_train, args.max_groups)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('train', train), ('probe', probe), ('holdout', holdout)]:
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as stream:
            for r in rows:
                stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    tokenizer.save_pretrained(out / 'tokenizer')
    clean_config = {k: v for k, v in vars(args).items() if k != 'func'}
    manifest = {'config': clean_config, 'model': args.model, 'model_revision': revision,
                'source_sha256': source_sha, 'corpus_revision': args.corpus_revision,
                'tokenizer_template_sha256': digest(tokenizer.chat_template), 'group_mapping': mapping,
                'tokenizer_hashes': {p.name: file_hash(p) for p in sorted((out / 'tokenizer').iterdir()) if p.is_file()},
                'eligible': len(records), 'rejected': len(rejected),
                'counts': {name: len(rows) for name, rows in [('train', train), ('probe', probe), ('holdout', holdout)]},
                'train_groups': dict(Counter(r['train_group'] for r in train)),
                'probe_groups': dict(Counter(r['train_group'] for r in probe)),
                'holdout_groups': dict(Counter(r['train_group'] for r in holdout)),
                'split_hashes': {name: file_hash(out / f'{name}.jsonl') for name in ('train', 'probe', 'holdout')},
                'train_input_tokens': sum(len(r['input_ids']) for r in train),
                'train_response_tokens': sum(r['response_tokens'] for r in train)}
    write_json(out / 'rejected.json', rejected)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'preview.json', [{k: r[k] for k in ('id', 'prompt', 'response', 'group', 'train_group')}
                                     for r in train[:12]])
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('Inspect preview.json, exclusions and group coverage before training.')


def per_example_loss(logits, labels):
    """Response-only mean CE, chunked to avoid fp32 [batch, sequence, vocab]."""
    import torch
    import torch.nn.functional as F
    result = []
    for row, target in zip(logits, labels):
        target = target[1:]
        positions = torch.where(target != -100)[0]
        if not positions.numel():
            raise ValueError('Sample has no response targets')
        total = row.new_zeros((), dtype=torch.float32)
        for indices in positions.split(128):
            total = total + F.cross_entropy(row[indices].float(), target[indices], reduction='sum')
        result.append(total / positions.numel())
    return torch.stack(result)


def train(args):
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              get_cosine_schedule_with_warmup, set_seed)
    if not torch.cuda.is_available():
        raise ValueError('Train on Kaggle/Colab CUDA GPU; CPU training disabled')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This pilot supports one process/GPU; do not launch with torchrun')
    if min(args.steps, args.batch_size, args.grad_accum, args.probe_every, args.save_every, args.log_every, args.rank) < 1:
        raise ValueError('steps, batch, accumulation and intervals must be positive')
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1:
        raise ValueError('Invalid learning rate/warmup')
    prepared, out = Path(args.prepared), Path(args.output)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for split, expected in manifest['split_hashes'].items():
        if file_hash(prepared / f'{split}.jsonl') != expected:
            raise ValueError(f'Prepared split changed: {split}')
    for name, expected in manifest['tokenizer_hashes'].items():
        if file_hash(prepared / 'tokenizer' / name) != expected:
            raise ValueError(f'Prepared tokenizer changed: {name}')
    data = {name: [r for _, r in read_records(prepared / f'{name}.jsonl')]
            for name in ('train', 'probe', 'holdout')}
    tokenizer = AutoTokenizer.from_pretrained(prepared / 'tokenizer')
    tokenizer.pad_token = tokenizer.eos_token
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    versions = {p: version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'numpy')}
    if args.four_bit:
        versions['bitsandbytes'] = version('bitsandbytes')
    config = {k: v for k, v in vars(args).items() if k not in ('output', 'resume', 'func')}
    identity = {'config': config, 'manifest_sha256': file_hash(prepared / 'manifest.json'),
                'source_sha256': digest([file_hash(Path(__file__)), file_hash(Path(__file__).with_name('distill_core.py'))]),
                'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'dtype': str(dtype)}
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output is not empty. Use --resume or choose a new run directory.')
    checkpoint = None
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if checkpoint.parent != out.resolve():
            raise ValueError('Resume in the original run output directory')
        if json.loads((checkpoint / 'identity.json').read_text()) != identity:
            raise ValueError('Resume configuration/data/source/environment differs')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'identity.json', identity)
    options = {'revision': manifest['model_revision'], 'torch_dtype': dtype,
               'device_map': {'': 0}, 'attn_implementation': 'sdpa'}
    if args.four_bit:
        options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], **options)
    model.config.use_cache = False
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint / 'adapter', is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type='CAUSAL_LM', r=args.rank, lora_alpha=args.rank * 2,
            lora_dropout=0., target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(args.steps * args.warmup_ratio), args.steps)
    scaler = torch.amp.GradScaler('cuda', enabled=not bf16)
    counts = manifest['train_groups']
    if args.mode != 'uniform' and len(counts) < 2:
        raise ValueError('Adaptive/group balancing requires at least two train groups; inspect grouping metadata')
    controller = GroupWeights(sorted(counts), counts, args.mode, args.mix,
                              args.temperature, args.ema, args.weight_floor, args.weight_ceiling)
    state = {'step': 0, 'epoch': 0, 'offset': 0, 'train_tokens': 0, 'response_tokens': 0,
             'probe_tokens': 0, 'holdout_tokens': 0, 'elapsed_seconds': 0., 'probe_seconds': 0.,
             'holdout_seconds': 0., 'optimizer_updates': 0, 'peak_vram_bytes': 0}
    if checkpoint:
        # Only load a checkpoint produced by this script from your own run.
        saved = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        controller.__dict__.update(saved['controller'])
        state = saved['state']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if state['step'] >= args.steps:
            raise ValueError('Checkpoint already reached requested steps')
        # Remove stale post-checkpoint log entries after a crash, without duplicating steps.
        log = out / 'metrics.jsonl'
        if log.exists():
            lines = [json.loads(s) for s in log.read_text().splitlines() if s.strip()]
            log.write_text(''.join(json.dumps(r) + '\n' for r in lines if r['step'] <= state['step']))
    start = time.monotonic()
    prior_elapsed = state['elapsed_seconds']
    torch.cuda.reset_peak_memory_stats()

    def batch(rows):
        length = max(len(r['input_ids']) for r in rows)
        ids = [r['input_ids'] + [tokenizer.pad_token_id] * (length - len(r['input_ids'])) for r in rows]
        labels = [r['labels'] + [-100] * (length - len(r['labels'])) for r in rows]
        mask = [[1] * len(r['input_ids']) + [0] * (length - len(r['input_ids'])) for r in rows]
        return {k: torch.tensor(v, device='cuda') for k, v in
                [('input_ids', ids), ('attention_mask', mask), ('labels', labels)]}

    def measure(split):
        torch.cuda.synchronize()
        begin = time.monotonic()
        totals, sizes = defaultdict(float), Counter()
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            for r in data[split]:
                b = batch([r])
                labels = b.pop('labels')
                output = model(**b, use_cache=False)
                loss = per_example_loss(output.logits, labels).item()
                totals[r['train_group']] += loss
                sizes[r['train_group']] += 1
                state[f'{split}_tokens'] += len(r['input_ids'])
                del output
        model.train()
        torch.cuda.synchronize()
        state[f'{split}_seconds'] += time.monotonic() - begin
        return {g: totals[g] / sizes[g] for g in totals}

    def log(value):
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': state['step'], **value}) + '\n')
        print(json.dumps({'step': state['step'], **value}), flush=True)

    def save():
        torch.cuda.synchronize()
        state['elapsed_seconds'] = prior_elapsed + time.monotonic() - start
        state['peak_vram_bytes'] = max(state['peak_vram_bytes'], torch.cuda.max_memory_allocated())
        target = out / f"checkpoint-{state['step']}"
        temp = out / f"checkpoint-{state['step']}.tmp"
        if target.exists() or temp.exists():
            raise ValueError(f'Checkpoint path exists: {target}; preserve or move it before resuming')
        temp.mkdir()
        model.save_pretrained(temp / 'adapter')
        tokenizer.save_pretrained(temp / 'adapter')
        write_json(temp / 'identity.json', identity)
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'controller': controller.__dict__, 'state': state,
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp / 'state.pt')
        temp.rename(target)
        write_json(out / 'latest.json', {'checkpoint': str(target.resolve()), 'state': state})

    if not checkpoint:
        initial = measure('probe')
        controller.update(initial)
        log({'event': 'initial_probe', 'probe_loss': initial, 'weights': controller.weights,
             'holdout_loss': measure('holdout')})
    model.train()
    order_epoch, order = None, None
    while state['step'] < args.steps:
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum):
            rows = []
            while len(rows) < args.batch_size:
                if order_epoch != state['epoch']:
                    order_epoch = state['epoch']
                    order = list(range(len(data['train'])))
                    random.Random(args.seed + state['epoch']).shuffle(order)
                take = min(args.batch_size - len(rows), len(order) - state['offset'])
                rows.extend(data['train'][i] for i in order[state['offset']:state['offset'] + take])
                state['offset'] += take
                if state['offset'] == len(order):
                    state['epoch'] += 1
                    state['offset'] = 0
            b = batch(rows)
            labels = b.pop('labels')
            with torch.autocast('cuda', dtype=dtype):
                output = model(**b, use_cache=False)
                losses = per_example_loss(output.logits, labels)
                weights = losses.new_tensor([controller.weights[r['train_group']] for r in rows])
                loss = (losses * weights).mean() / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; stop and lower LR/check data/precision')
            scaler.scale(loss).backward()
            micro_losses.append(losses.detach().mean().item())
            state['train_tokens'] += sum(len(r['input_ids']) for r in rows)
            state['response_tokens'] += sum(r['response_tokens'] for r in rows)
            del output, losses, loss
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            state['optimizer_updates'] += 1
        state['step'] += 1
        if state['step'] % args.log_every == 0 or state['step'] == args.steps:
            log({'event': 'train', 'response_ce': sum(micro_losses) / len(micro_losses),
                 'grad_norm': float(grad_norm), 'learning_rate': scheduler.get_last_lr()[0],
                 'train_tokens': state['train_tokens'], 'response_tokens': state['response_tokens'],
                 'optimizer_updates': state['optimizer_updates']})
        if state['step'] % args.probe_every == 0 or state['step'] == args.steps:
            losses = measure('probe')
            if state['step'] >= args.warmup_steps:
                controller.update(losses)
            log({'event': 'probe', 'probe_loss': losses, 'weights': controller.weights,
                 'holdout_loss': measure('holdout')})
        if state['step'] % args.save_every == 0 or state['step'] == args.steps:
            save()
    model.save_pretrained(out / 'adapter')
    tokenizer.save_pretrained(out / 'adapter')
    write_json(out / 'summary.json', {'mode': args.mode, 'model': manifest['model'], 'state': state,
                                    'weights': controller.weights,
                                    'note': 'Teacher-response CE is not benchmark accuracy/F1. Run SentiBench separately.'})


def export(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    run = Path(args.run)
    identity = json.loads((run / 'identity.json').read_text())
    prepared = Path(identity['config']['prepared'])
    manifest = json.loads((prepared / 'manifest.json').read_text())
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise ValueError('Export target must be empty')
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], revision=manifest['model_revision'],
                                               torch_dtype=torch.float32, device_map={'': 'cpu'})
    model = PeftModel.from_pretrained(model, run / 'adapter').merge_and_unload()
    model.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(run / 'adapter').save_pretrained(target)
    write_json(target / 'distillation_run.json', identity)
    print('Merged model:', target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    a = commands.add_parser('audit')
    a.add_argument('--file', required=True)
    a.add_argument('--group_field', default='cluster_id')
    a.add_argument('--examples', type=int, default=5)
    a.add_argument('--output', required=True)
    a.set_defaults(func=audit)
    p = commands.add_parser('prepare')
    p.add_argument('--file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--model_revision', default='main')
    p.add_argument('--corpus_revision', required=True)
    p.add_argument('--prompt_fields', required=True, help='Comma-separated fields in verified order')
    p.add_argument('--response_field', required=True)
    p.add_argument('--group_field', required=True)
    p.add_argument('--source_field', required=True, help='Original input text for leakage-safe grouped splitting')
    p.add_argument('--mapping_confirmed', action='store_true')
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--max_train', type=int, default=5000, help='0 = all eligible training records')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dev_fraction', type=float, default=0.1)
    p.add_argument('--probe_per_group', type=int, default=4)
    p.add_argument('--holdout_per_group', type=int, default=4)
    p.add_argument('--min_group_train', type=int, default=8)
    p.add_argument('--max_groups', type=int, default=16)
    p.set_defaults(func=prepare)
    t = commands.add_parser('train')
    t.add_argument('--prepared', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--mode', choices=['uniform', 'balanced', 'loss', 'progress'], default='uniform')
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--batch_size', type=int, default=1)
    t.add_argument('--grad_accum', type=int, default=8)
    t.add_argument('--learning_rate', type=float, default=1e-4)
    t.add_argument('--rank', type=int, default=16)
    t.add_argument('--warmup_ratio', type=float, default=0.05)
    t.add_argument('--warmup_steps', type=int, default=25, help='Before adaptive updates')
    t.add_argument('--probe_every', type=int, default=25)
    t.add_argument('--save_every', type=int, default=100)
    t.add_argument('--log_every', type=int, default=10)
    t.add_argument('--mix', type=float, default=0.5)
    t.add_argument('--temperature', type=float, default=0.05)
    t.add_argument('--ema', type=float, default=0.5)
    t.add_argument('--weight_floor', type=float, default=0.5)
    t.add_argument('--weight_ceiling', type=float, default=2.)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--four_bit', action='store_true')
    t.add_argument('--resume', help='Own checkpoint directory; restores optimizer/scaler/RNG/controller/cursor')
    t.set_defaults(func=train)
    e = commands.add_parser('export')
    e.add_argument('--run', required=True)
    e.add_argument('--output', required=True)
    e.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
