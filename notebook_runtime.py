"""Shared, memory-conscious evaluation support for hosted notebooks."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import hashlib
import json
import os
from pathlib import Path
import random
import warnings


def add_runtime_arguments(parser):
    parser.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0: all; otherwise first N test examples')
    parser.add_argument('--demo_split', choices=['train', 'val'], default='train')
    parser.add_argument('--eval_split', choices=['test', 'val'], default='test')
    parser.add_argument('--sample_strategy', choices=['first', 'random'], default='first')
    parser.add_argument('--model_revision', default=None)
    parser.add_argument('--measure_inference', action='store_true', help='Forward/generation timing after one warm-up batch')
    parser.add_argument('--max_eval_seconds', type=float, default=0., help='0: unlimited; save cache and stop between batches')
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--resume_file', help='Atomic prediction checkpoint; use a persistent output path')
    parser.add_argument('--adapter', default=None, help='Path to PEFT adapter checkpoint directory')


def prepare_runtime(args):
    import torch
    import sys
    evaluator = Path(sys.argv[0]).name
    if getattr(args, 'eval_split', 'test') != 'test' and evaluator not in ('icl_multifaced_kaggle.py', 'fsa_kaggle.py'):
        raise ValueError('--eval_split is supported by *_kaggle.py evaluators only')
    if getattr(args, 'measure_inference', False) and evaluator not in ('icl_multifaced_kaggle.py', 'fsa_kaggle.py'):
        raise ValueError('--measure_inference requires a *_kaggle.py evaluator')
    if getattr(args, 'eval_split', None) == 'test' and (args.max_samples != 0 or args.data_prop != 1):
        raise ValueError('Final test must use the full dataset: max_samples=0, data_prop=1. Use val for subsets.')
    if getattr(args, 'max_eval_seconds', 0) < 0:
        raise ValueError('max_eval_seconds must be nonnegative')
    if args.eval_batch_size < 1 or args.k_shot < 0 or args.max_samples < 0:
        raise ValueError('batch_size must be positive; k_shot and max_samples must be nonnegative')
    if args.max_seq_length != -1 and args.max_seq_length < 1:
        raise ValueError('max_seq_length must be -1 or a positive token count')
    if not 0 < args.data_prop <= 1 or args.max_new_tokens < 1:
        raise ValueError('data_prop must be in (0, 1]; max_new_tokens must be positive')
    if args.model_name_or_path:
        model_path = args.model_name_or_path
    else:
        json_file = Path(__file__).with_name('model_name_kaggle.json')
        if not json_file.exists():
            json_file = Path(__file__).with_name('model_name.json')
        mapping = json.loads(json_file.read_text())
        model_path = mapping.get(args.model_version, args.model_version)
        if not model_path:
            raise ValueError('Provide --model_name_or_path or --model_version')
        if model_path.startswith('.'):
            model_path = str(Path(__file__).parent / model_path)
    args.model_name_or_path = model_path
    if args.load_in_4bit and not torch.cuda.is_available():
        raise ValueError('4-bit evaluation requires an available CUDA GPU')
    dtype = args.dtype
    if dtype == 'auto':
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
        dtype = ('bfloat16' if native_bf16 else 'float16') if torch.cuda.is_available() else 'float32'
    if dtype == 'bfloat16' and (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported()):
        raise ValueError('This GPU does not support BF16; use --dtype auto or float16')
    if not torch.cuda.is_available() and dtype == 'float16':
        raise ValueError('Use float32 for CPU evaluation')
    args.resolved_dtype = dtype
    args.runtime_device = {'name': torch.cuda.get_device_name(0),
        'capability': list(torch.cuda.get_device_capability(0)),
        'cuda_version': torch.version.cuda} if torch.cuda.is_available() else {'name': 'CPU'}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f'Runtime: {"CUDA" if torch.cuda.is_available() else "CPU"}, dtype={dtype}, 4bit={args.load_in_4bit}')
    return model_path


def _disable_torchao():
    import sys
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.dispatch_torchao = lambda *args, **kwargs: None
    except Exception:
        pass
    for mod in list(sys.modules.values()):
        if hasattr(mod, 'is_torchao_available'):
            try:
                mod.is_torchao_available = lambda: False
            except Exception:
                pass


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    options = dict(torch_dtype=getattr(torch, args.resolved_dtype), low_cpu_mem_usage=True)
    if getattr(args, 'model_revision', None):
        options['revision'] = args.model_revision
    if torch.cuda.is_available():
        # One GPU avoids accidental CPU offload or combining incompatible notebook GPUs.
        options['device_map'] = {'': 0}
    if args.load_in_4bit:
        options['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=getattr(torch, args.resolved_dtype))
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **options)
    args.resolved_model_revision = getattr(model.config, '_commit_hash', None)
    if getattr(args, 'adapter', None):
        _disable_torchao()
        from peft import PeftModel
        print(f'Loading PEFT adapter from {args.adapter}')
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model


def select_examples(raw, args):
    for index, example in enumerate(raw['test']):
        example['_evaluation_id'] = index
    count = max(1, int(len(raw['test']) * args.data_prop)) if raw['test'] else 0
    if args.max_samples:
        count = min(count, args.max_samples)
    if getattr(args, 'sample_strategy', 'first') == 'random':
        raw['test'] = random.Random(args.seed).sample(raw['test'], count)
    else:
        raw['test'] = raw['test'][:count]
    if not raw['test']:
        raise ValueError('Test split is empty')
    if args.k_shot > len(raw['demo']):
        raise ValueError(f'k_shot={args.k_shot} exceeds demo pool ({len(raw["demo"])})')
    return raw


def configure_tokenizer(tokenizer, model, args, output_tokens):
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'  # Keep the final question and Label: suffix.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer needs a pad or EOS token')
    limits = [getattr(tokenizer, 'model_max_length', 10**9),
              getattr(model.config, 'max_position_embeddings', 10**9)]
    context = min(int(v) for v in limits if v is not None)
    available = context - output_tokens
    if available < 1:
        raise ValueError('Generation budget exceeds the model context window')
    if args.max_seq_length == -1:
        args.max_seq_length = available
    else:
        args.max_seq_length = min(args.max_seq_length, available)


def limit_prompts(examples, tokenizer, max_length):
    """Truncate demonstrations first; reject loss of the actual test sentence."""
    truncated = 0
    for example in examples:
        text = example['prompts']
        # The final prompt suffix contains the complete test input.
        suffix = f"Sentence: {example['sentence']}\nLabel:"
        if len(tokenizer(suffix, add_special_tokens=False)['input_ids']) + tokenizer.num_special_tokens_to_add() > max_length:
            raise ValueError('Test sentence exceeds max_seq_length. Increase the limit; refusing to silently drop test text.')
        ids = tokenizer(text, add_special_tokens=True)['input_ids']
        if len(ids) > max_length:
            truncated += 1
    if truncated:
        warnings.warn(f'{truncated}/{len(examples)} prompts exceed {max_length} tokens. Left truncation removes instructions/demos; results differ from untruncated evaluation.')
    return truncated


def run_metadata(args, raw, truncated):
    from importlib.metadata import version
    settings = {k: getattr(args, k) for k in (
        'model_name_or_path', 'k_shot', 'max_seq_length', 'max_samples', 'data_prop',
        'demo_split', 'resolved_dtype', 'load_in_4bit', 'max_new_tokens', 'eval_batch_size')}
    settings.update({k: getattr(args, k, None) for k in
                     ('eval_split', 'sample_strategy', 'model_revision', 'adapter', 'measure_inference')})
    adapter_hashes = None
    if getattr(args, 'adapter', None):
        adapter_path = Path(args.adapter)
        adapter_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(adapter_path.iterdir()) if p.suffix in ('.json', '.safetensors', '.bin')}
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    source_files = [Path(__file__), Path(__file__).with_name('fsa.py'),
                    Path(__file__).with_name('icl_multifaced.py'), Path(__file__).with_name('utils.py'),
                    Path(__file__).with_name('fsa_kaggle.py'), Path(__file__).with_name('icl_multifaced_kaggle.py')]
    source_files = [p for p in source_files if p.is_file()]
    source_files += sorted(Path(__file__).parent.glob('parse_utils/*.py'))
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()
    demo_ids = [example.get('demo_ids') for example in raw['test']]
    demo_hash = hashlib.sha256(json.dumps(demo_ids, sort_keys=True).encode()).hexdigest()
    return dict(experiment=args.subname, dataset=args.dataset, seed=args.seed,
                settings=settings, data_sha256=fingerprint, num_samples=len(raw['test']),
                source_sha256=source_hash, model_revision=args.resolved_model_revision, adapter_sha256=adapter_hashes,
                demo_manifest_sha256=demo_hash,
                runtime_device=getattr(args, 'runtime_device', None),
                truncated_prompts=truncated,
                versions={p: version(p) for p in ('torch', 'transformers', 'accelerate', 'numpy', 'scikit-learn')})


class PredictionCache:
    """Atomic batch checkpoints; validate data/settings before reusing predictions."""
    def __init__(self, path, metadata):
        self.path = Path(path) if path else None
        self.metadata = metadata
        self.entries = {}
        self.performance = {'seconds': 0., 'examples': 0, 'generated_tokens': 0,
                            'batch_seconds': [], 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved['metadata'] != metadata:
                raise ValueError('Checkpoint settings/data/environment differ. Choose a new resume_file or restore the original configuration.')
            self.entries = saved['entries']
            self.performance = saved.get('performance', self.performance)
            print(f'Resuming {len(self.entries)} cached predictions')

    @staticmethod
    def key(example):
        return hashlib.sha256(json.dumps([example.get('_evaluation_id'), example['prompts']]).encode()).hexdigest()

    def get(self, example):
        return self.entries.get(self.key(example))

    def put(self, example, value):
        self.entries[self.key(example)] = value

    def measure(self, seconds, examples, generated_tokens=0, allocated=0, reserved=0):
        p = self.performance
        p['seconds'] += seconds
        p['examples'] += examples
        p['generated_tokens'] += generated_tokens
        p['batch_seconds'].append(seconds)
        p['peak_allocated_bytes'] = max(p['peak_allocated_bytes'], allocated)
        p['peak_reserved_bytes'] = max(p['peak_reserved_bytes'], reserved)

    def measurement_summary(self, kind, batch_size):
        import math
        p = self.performance
        times = sorted(p['batch_seconds'])
        return {'scope': kind + '; excludes loading/tokenization/copies/scoring; one warm-up batch per invocation.',
            'measured_examples': p['examples'], 'measured_batches': len(times), 'batch_size': batch_size,
            'inference_seconds': p['seconds'],
            'amortized_ms_per_example': p['seconds'] * 1000 / p['examples'] if p['examples'] else None,
            'examples_per_second': p['examples'] / p['seconds'] if p['seconds'] else None,
            'p95_batch_latency_ms': times[math.ceil(.95 * len(times)) - 1] * 1000 if times else None,
            'generated_tokens': p['generated_tokens'],
            'generated_tokens_per_second': p['generated_tokens'] / p['seconds'] if p['seconds'] and p['generated_tokens'] else None,
            'peak_allocated_bytes': p['peak_allocated_bytes'], 'peak_reserved_bytes': p['peak_reserved_bytes']}

    def flush(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + '.tmp')
            temp.write_text(json.dumps({'metadata': self.metadata, 'entries': self.entries,
                                       'performance': self.performance}), encoding='utf-8')
            os.replace(temp, self.path)


def save_result_kaggle(output_dir, exper_name, f1, acc, metadata=None):
    result = (exper_name, {'f1': round(f1, 4), 'acc': round(acc, 4) if acc is not None else None})
    if metadata is not None:
        result[1]['metadata'] = metadata
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'result.txt'), 'a', encoding='utf-8-sig') as f:
        f.write(json.dumps(result) + '\n')
