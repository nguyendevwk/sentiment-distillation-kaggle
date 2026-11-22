"""Paired Bootstrap Test for Statistical Significance on SentiBench.

Compares two prediction JSONL files matching on record ID across 1,000 resamples.
Outputs delta, 95% confidence intervals, and two-sided p-value.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys


def read_predictions(path):
    rows = {}
    with Path(path).open('r', encoding='utf-8') as stream:
        for idx, line in enumerate(stream):
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = item.get('id', f'row_{idx}')
            rows[record_id] = item
    return rows


def calculate_sample_metric(gold, pred):
    """Normalized match between gold and pred."""
    if isinstance(gold, (list, tuple)):
        gold_set = set(tuple(x) if isinstance(x, (list, tuple)) else str(x) for x in gold)
        pred_set = set(tuple(x) if isinstance(x, (list, tuple)) else str(x) for x in (pred or []))
        if not gold_set and not pred_set:
            return 1.0
        if not gold_set or not pred_set:
            return 0.0
        intersection = len(gold_set.intersection(pred_set))
        precision = intersection / len(pred_set)
        recall = intersection / len(gold_set)
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return 1.0 if str(gold).strip().lower() == str(pred).strip().lower() else 0.0


def paired_bootstrap(baseline_path, treatment_path, n_resamples=1000, seed=42):
    base_preds = read_predictions(baseline_path)
    treat_preds = read_predictions(treatment_path)

    common_ids = sorted(set(base_preds.keys()).intersection(set(treat_preds.keys())))
    if not common_ids:
        raise ValueError("No common prediction IDs found between baseline and treatment")

    base_scores = []
    treat_scores = []
    for cid in common_ids:
        b_item = base_preds[cid]
        t_item = treat_preds[cid]
        gold = b_item.get('gold', t_item.get('gold'))
        b_score = calculate_sample_metric(gold, b_item.get('pred'))
        t_score = calculate_sample_metric(gold, t_item.get('pred'))
        base_scores.append(b_score)
        treat_scores.append(t_score)

    n = len(common_ids)
    obs_base = statistics.mean(base_scores)
    obs_treat = statistics.mean(treat_scores)
    obs_delta = obs_treat - obs_base

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        sample_indices = [rng.randrange(n) for _ in range(n)]
        b_sample_mean = statistics.mean(base_scores[i] for i in sample_indices)
        t_sample_mean = statistics.mean(treat_scores[i] for i in sample_indices)
        deltas.append(t_sample_mean - b_sample_mean)

    deltas.sort()
    ci_lower = deltas[int(0.025 * n_resamples)]
    ci_upper = deltas[int(0.975 * n_resamples)]

    # Two-sided p-value under null hypothesis that true delta is 0
    mean_delta_boot = statistics.mean(deltas)
    shifted_deltas = [d - mean_delta_boot for d in deltas]
    p_value = sum(1 for d in shifted_deltas if abs(d) >= abs(obs_delta)) / float(n_resamples)

    return {
        'n_samples': n,
        'baseline_mean': obs_base,
        'treatment_mean': obs_treat,
        'observed_delta': obs_delta,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'p_value': p_value,
        'statistically_significant_005': p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help="Baseline predictions .jsonl")
    parser.add_argument('--treatment', required=True, help="Treatment predictions .jsonl")
    parser.add_argument('--n-resamples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', help="Output path for JSON report")
    args = parser.parse_args()

    results = paired_bootstrap(args.baseline, args.treatment, args.n_resamples, args.seed)

    print("\n# Paired Bootstrap Significance Test Results\n")
    print(f"| Metric | Baseline | Treatment | Delta (Treatment - Baseline) | 95% CI | p-value | Significant (p < 0.05) |")
    print(f"|---|---:|---:|---:|---:|---:|:---:|")
    print(f"| Score | {results['baseline_mean']*100:.2f}% | {results['treatment_mean']*100:.2f}% | "
          f"{results['observed_delta']*100:+.2f}% | [{results['ci_95_lower']*100:+.2f}%, {results['ci_95_upper']*100:+.2f}%] | "
          f"{results['p_value']:.4f} | {'YES' if results['statistically_significant_005'] else 'NO'} |\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Saved results to {args.output}")


if __name__ == '__main__':
    main()
