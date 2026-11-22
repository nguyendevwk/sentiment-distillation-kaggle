"""Kaggle evaluation suite: evaluate and compare Base Model (Before) vs Distilled Model (After LoRA).

Runs evaluation in separate subprocesses to prevent CUDA memory fragmentation and OOM.
Generates side-by-side comparison tables and saves evaluation_comparison.json / .md.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_single_eval(script_name, model, adapter, dataset, k_shot, batch_size,
                    max_seq_length, max_samples, output_dir, seed):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subname = 'eval_finetuned' if adapter else 'eval_base'

    cmd = [
        sys.executable,
        str(Path(__file__).parent / script_name),
        '--dataset', dataset,
        '--subname', subname,
        '--seed', str(seed),
        '--model_name_or_path', model,
        '--k_shot', str(k_shot),
        '--eval_batch_size', str(batch_size),
        '--max_seq_length', str(max_seq_length),
        '--output_dir', str(output_dir),
    ]
    if adapter:
        cmd.extend(['--adapter', str(adapter)])
    if max_samples > 0:
        cmd.extend(['--max_samples', str(max_samples)])

    print(f"\n{'='*70}")
    target_desc = f"{model} + Adapter ({adapter})" if adapter else f"Base Model ({model})"
    print(f"Running evaluation: {dataset} | {target_desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start_time = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_time

    # Read latest result from result.txt
    result_file = output_dir / 'result.txt'
    if not result_file.exists():
        raise RuntimeError(f"Expected result file not found: {result_file}")

    last_record = None
    with open(result_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                try:
                    last_record = json.loads(line)
                except Exception:
                    pass

    if not last_record:
        raise RuntimeError(f"Could not parse valid result record from {result_file}")

    metrics = last_record[1]
    return {
        'f1': metrics.get('f1', 0.0),
        'accuracy': metrics.get('acc', 0.0),
        'elapsed_seconds': round(elapsed, 2),
        'experiment_name': last_record[0]
    }


def format_delta(base_val, ft_val):
    delta = ft_val - base_val
    sign = '+' if delta > 0 else ''
    icon = '▲' if delta > 0.0005 else ('▼' if delta < -0.0005 else '=')
    return f"{ft_val*100:.2f}% ({sign}{delta*100:.2f}% {icon})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct', help='Base model HF ID or path')
    parser.add_argument('--adapter', default=None, help='Path to trained PEFT LoRA adapter directory')
    parser.add_argument('--datasets', default='sc/imdb,sc/sst2', help='Comma-separated datasets to evaluate')
    parser.add_argument('--k_shot', type=int, default=4, help='Few-shot demonstrations')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Context window cutoff')
    parser.add_argument('--max_samples', type=int, default=0, help='Max test examples (0 for all)')
    parser.add_argument('--output_dir', default='./output/evaluation_comparison', help='Output report directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_base', action='store_true', help='Skip base model evaluation if already recorded')
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    # Guard against Kaggle read-only /kaggle/input filesystem
    if str(out_dir).startswith('/kaggle/input'):
        parts = out_dir.parts
        # Redirect to /kaggle/working/output/...
        out_dir = Path('/kaggle/working') / 'output' / out_dir.name
        print(f"[Kaggle Guard] Output path was inside read-only /kaggle/input. Auto-redirected to writable: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]

    model_name = args.model
    if args.adapter:
        adapter_cfg_file = Path(args.adapter) / 'adapter_config.json'
        if adapter_cfg_file.exists():
            try:
                cfg = json.loads(adapter_cfg_file.read_text())
                detected_model = cfg.get('base_model_name_or_path')
                if detected_model and detected_model != model_name:
                    print(f"\n{'*'*70}")
                    print(f"[Auto-Align] Tự động phát hiện base model gốc: '{detected_model}'")
                    print(f"[Auto-Align] Đã tự động thay thế '{model_name}' -> '{detected_model}' để tránh lỗi size mismatch!")
                    print(f"{'*'*70}\n")
                    model_name = detected_model
            except Exception as e:
                print(f"[Warning] Không thể đọc adapter_config.json: {e}")

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'base_model': model_name,
        'adapter': args.adapter,
        'k_shot': args.k_shot,
        'max_samples': args.max_samples,
        'results': {}
    }

    base_dir = out_dir / 'raw_base'
    ft_dir = out_dir / 'raw_finetuned'

    for ds in datasets:
        script = 'fsa_kaggle.py' if ds.startswith('absa/') else 'icl_multifaced_kaggle.py'
        report['results'][ds] = {}

        # 1. Base Model evaluation
        base_res = None
        if not args.skip_base:
            base_res = run_single_eval(
                script_name=script,
                model=model_name,
                adapter=None,
                dataset=ds,
                k_shot=args.k_shot,
                batch_size=args.eval_batch_size,
                max_seq_length=args.max_seq_length,
                max_samples=args.max_samples,
                output_dir=base_dir,
                seed=args.seed
            )
            report['results'][ds]['base'] = base_res

        # 2. Fine-tuned Model evaluation (if adapter specified)
        if args.adapter:
            ft_res = run_single_eval(
                script_name=script,
                model=model_name,
                adapter=args.adapter,
                dataset=ds,
                k_shot=args.k_shot,
                batch_size=args.eval_batch_size,
                max_seq_length=args.max_seq_length,
                max_samples=args.max_samples,
                output_dir=ft_dir,
                seed=args.seed
            )
            report['results'][ds]['finetuned'] = ft_res

            if base_res:
                report['results'][ds]['delta_f1'] = round(ft_res['f1'] - base_res['f1'], 4)
                report['results'][ds]['delta_accuracy'] = round(ft_res['accuracy'] - base_res['accuracy'], 4)

    # 3. Print & Save Comparison Table
    print("\n" + "="*84)
    print("           SENTIMENT DISTILLATION: BEFORE vs AFTER EVALUATION REPORT")
    print("="*84)
    print(f"Base Model:         {model_name}")
    print(f"Adapter Checkpoint: {args.adapter or '<None>'}")
    print(f"Few-shot (k_shot):  {args.k_shot}")
    print(f"Test samples:       {'All' if args.max_samples == 0 else args.max_samples}")
    print("-"*84)
    header = f"{'Dataset':<20} | {'Metric':<10} | {'Base Model (Before)':<20} | {'Distilled (After)':<26}"
    print(header)
    print("-"*84)

    lines = []
    lines.append(f"# Báo cáo Đối chứng Đánh giá Trước vs Sau Chưng cất (Sentiment Distillation)\n")
    lines.append(f"- **Mô hình Gốc (Base)**: `{model_name}`")
    lines.append(f"- **LoRA Adapter**: `{args.adapter or 'N/A'}`")
    lines.append(f"- **Số mẫu Demo (k-shot)**: `{args.k_shot}`")
    lines.append(f"- **Thời gian đánh giá**: `{report['timestamp']}`\n")
    lines.append(f"| Tác vụ (Dataset) | Chỉ số | Mô hình Gốc (Trước) | Sau Finetuning (Adapter) | Mức độ cải thiện (Delta) |")
    lines.append(f"|---|---|---|---|---|")

    for ds, data in report['results'].items():
        base = data.get('base')
        ft = data.get('finetuned')

        if base and ft:
            f1_str = format_delta(base['f1'], ft['f1'])
            acc_str = format_delta(base['accuracy'], ft['accuracy'])
            print(f"{ds:<20} | {'Macro F1':<10} | {base['f1']*100:>6.2f}%              | {f1_str:<26}")
            print(f"{ds:<20} | {'Accuracy':<10} | {base['accuracy']*100:>6.2f}%              | {acc_str:<26}")
            print("-"*84)
            d_f1 = ft['f1'] - base['f1']
            d_acc = ft['accuracy'] - base['accuracy']
            lines.append(f"| `{ds}` | Macro F1 | {base['f1']*100:.2f}% | {ft['f1']*100:.2f}% | **{'+' if d_f1>=0 else ''}{d_f1*100:.2f}%** |")
            lines.append(f"| `{ds}` | Accuracy | {base['accuracy']*100:.2f}% | {ft['accuracy']*100:.2f}% | **{'+' if d_acc>=0 else ''}{d_acc*100:.2f}%** |")
        elif base:
            print(f"{ds:<20} | {'Macro F1':<10} | {base['f1']*100:>6.2f}%              | {'N/A':<26}")
            print(f"{ds:<20} | {'Accuracy':<10} | {base['accuracy']*100:>6.2f}%              | {'N/A':<26}")
            print("-"*84)
            lines.append(f"| `{ds}` | Macro F1 | {base['f1']*100:.2f}% | N/A | N/A |")
            lines.append(f"| `{ds}` | Accuracy | {base['accuracy']*100:.2f}% | N/A | N/A |")

    print("="*84 + "\n")

    # Save JSON and Markdown
    json_path = out_dir / 'evaluation_comparison.json'
    md_path = out_dir / 'evaluation_comparison.md'
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Detailed comparison saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}\n")


if __name__ == '__main__':
    main()
