"""Validate the evaluator's dependency closure without importing unrelated host apps."""
import argparse
import importlib
from importlib.metadata import PackageNotFoundError, distribution, version
import json
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement


def check(requirements_path, four_bit=False, require_gpu=False):
    requirements = [Requirement(line.strip()) for line in Path(requirements_path).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
    requirements.extend([Requirement('torch>=2.2,<3'), Requirement('packaging>=23')])
    if four_bit:
        bnb_version = '0.46.1' if sys.version_info[:2] >= (3, 13) else '0.45.5'
        requirements.append(Requirement(f'bitsandbytes=={bnb_version}'))
    pending = [(r, 'notebook requirements') for r in requirements]
    visited, errors = set(), []
    while pending:
        requirement, parent = pending.pop()
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        try:
            dist = distribution(requirement.name)
        except PackageNotFoundError:
            errors.append(f'{parent}: missing {requirement}')
            continue
        if requirement.specifier and not requirement.specifier.contains(dist.version, prereleases=True):
            errors.append(f'{parent}: needs {requirement}, found {dist.version}')
        key = dist.metadata['Name'].lower().replace('_', '-')
        if key in visited:
            continue
        visited.add(key)
        pending.extend((Requirement(r), dist.metadata['Name']) for r in dist.requires or [])
    if errors:
        raise RuntimeError('Dependency conflicts:\n' + '\n'.join(sorted(set(errors))))
    for module in ('torch', 'transformers', 'accelerate', 'numpy', 'scipy', 'sklearn', 'nltk', 'sentencepiece',
                   'fsa', 'icl_multifaced'):
        importlib.import_module(module)
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Enable a GPU accelerator, then rerun.')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith('cuda') else torch.float32
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16,
                                     n_layer=1, n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    model = model.to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        inputs = torch.tensor([[1, 3, 4]], device=device)
        logits = model(inputs, attention_mask=torch.ones_like(inputs)).logits
        assert torch.isfinite(logits).all()
        generated = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=1, do_sample=False)
        assert generated.shape[-1] == 4
    if four_bit:
        import bitsandbytes as bnb
        layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=dtype, quant_type='nf4').to(device)
        with torch.inference_mode():
            value = layer(torch.randn(2, 64, device=device, dtype=dtype))
            assert torch.isfinite(value).all()
        del layer
    report = {
        'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable,
        'torch': version('torch'), 'cuda_build': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'dtype': str(dtype), 'four_bit_tested': four_bit,
        'packages': {name: version(name) for name in sorted(visited)},
        'dependency_check': 'passed', 'tiny_model_forward_and_generation': 'passed',
    }
    print(json.dumps(report, indent=2))
    del model, logits, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    profile = 'requirements-notebook-py313.txt' if sys.version_info[:2] >= (3, 13) else 'requirements-notebook.txt'
    parser.add_argument('--requirements', default=str(Path(__file__).with_name(profile)))
    parser.add_argument('--four-bit', action='store_true')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--report', default='environment-report.json')
    args = parser.parse_args()
    report = check(args.requirements, args.four_bit, args.require_gpu)
    Path(args.report).write_text(json.dumps(report, indent=2))
