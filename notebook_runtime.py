"""Shared, memory-conscious evaluation support for hosted notebooks."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import hashlib
import json
import os
from pathlib import Path
import warnings


def add_runtime_arguments(parser):
    parser.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0: all; otherwise first N test examples')
    parser.add_argument('--demo_split', choices=['train', 'val'], default='train')
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--resume_file', help='Atomic prediction checkpoint; use a persistent output path')
    parser.add_argument('--adapter', default=None, help='Path to PEFT adapter checkpoint directory')


def prepare_runtime(args):
    import torch
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
        dtype = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if torch.cuda.is_available() else 'float32'
    if dtype == 'bfloat16' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('This GPU does not support BF16; use --dtype auto or float16')
    if not torch.cuda.is_available() and dtype == 'float16':
        raise ValueError('Use float32 for CPU evaluation')
    args.resolved_dtype = dtype
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
    count = max(1, int(len(raw['test']) * args.data_prop)) if raw['test'] else 0
    if args.max_samples:
        count = min(count, args.max_samples)
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
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    source_files = [Path(__file__), Path(__file__).with_name('fsa.py'),
                    Path(__file__).with_name('icl_multifaced.py'), Path(__file__).with_name('utils.py')]
    source_files += sorted(Path(__file__).parent.glob('parse_utils/*.py'))
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()
    return dict(experiment=args.subname, dataset=args.dataset, seed=args.seed,
                settings=settings, data_sha256=fingerprint, num_samples=len(raw['test']),
                source_sha256=source_hash, model_revision=args.resolved_model_revision,
                truncated_prompts=truncated,
                versions={p: version(p) for p in ('torch', 'transformers', 'accelerate', 'numpy', 'scikit-learn')})


class PredictionCache:
    """Atomic batch checkpoints; validate data/settings before reusing predictions."""
    def __init__(self, path, metadata):
        self.path = Path(path) if path else None
        self.metadata = metadata
        self.entries = {}
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved['metadata'] != metadata:
                raise ValueError('Checkpoint settings/data/environment differ. Choose a new resume_file or restore the original configuration.')
            self.entries = saved['entries']
            print(f'Resuming {len(self.entries)} cached predictions')

    @staticmethod
    def key(example):
        return hashlib.sha256(example['prompts'].encode()).hexdigest()

    def get(self, example):
        return self.entries.get(self.key(example))

    def put(self, example, value):
        self.entries[self.key(example)] = value

    def flush(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + '.tmp')
            temp.write_text(json.dumps({'metadata': self.metadata, 'entries': self.entries}), encoding='utf-8')
            os.replace(temp, self.path)


def save_result_kaggle(output_dir, exper_name, f1, acc, metadata=None):
    result = (exper_name, {'f1': round(f1, 4), 'acc': round(acc, 4)})
    if metadata is not None:
        result[1]['metadata'] = metadata
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'result.txt'), 'a', encoding='utf-8-sig') as f:
        f.write(json.dumps(result) + '\n')
