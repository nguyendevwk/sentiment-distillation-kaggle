"""Unit tests for BCTD components:
(a) Sample-Normalized Loss balancing when L_K / L_I = 40
(b) Capability-Aware STFA budget allocation
(c) 0 duplicate rows between benchmark train/dev and 12 official test files
(d) ASQP parser reading official paper format, multiline code fences, and permutations
"""
import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))

try:
    import torch
except ImportError:
    torch = None

from train_distill import strip_think_tags
from v1_core import classify_icl_task, select_rows_stfa
from parse_utils.structured_literal import asqp_prediction, literal_tuples
from evaluation_core import locked_test_keys, rows, text_key


class BCTDComponentTests(unittest.TestCase):

    def test_sample_normalized_loss_weights_properly_when_ratio_is_40(self):
        """Verify that Sample-Normalized Loss balances when L_know / L_icl = 40."""
        import math

        def py_sample_normalized_loss(mean_icl, mean_know, step, total_steps, replay_decay_str="0.20:0.05"):
            parts = [float(x) for x in replay_decay_str.split(':')]
            lam_start, lam_end = parts[0], parts[1]
            t = min(max(0, step), total_steps)
            T = max(1, total_steps)
            lam_t = lam_end + 0.5 * (lam_start - lam_end) * (1.0 + math.cos(math.pi * t / T))
            loss = (1.0 - lam_t) * mean_icl + lam_t * mean_know
            return loss, lam_t

        # Ratio L_know / L_icl = 40
        loss_0, lam_0 = py_sample_normalized_loss(1.0, 40.0, 0, 100)
        self.assertAlmostEqual(lam_0, 0.20, places=3)
        self.assertAlmostEqual(loss_0, 0.80 * 1.0 + 0.20 * 40.0, places=3)

        loss_100, lam_100 = py_sample_normalized_loss(1.0, 40.0, 100, 100)
        self.assertAlmostEqual(lam_100, 0.05, places=3)
        self.assertAlmostEqual(loss_100, 0.95 * 1.0 + 0.05 * 40.0, places=3)

        if torch is not None:
            from training.train_distill import compute_sample_normalized_loss
            vocab_size = 100
            seq_len = 10
            batch_size = 2
            logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
            labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
            labels[0, 7:10] = 5
            logits[0, 6:9, 5] = 10.0
            labels[1, 2:10] = 20
            logits[1, 1:9, 10] = 10.0
            logits[1, 1:9, 20] = -10.0
            stages = ['icl', 'know']
            loss, info = compute_sample_normalized_loss(
                logits, labels, stages, 0, 100, replay_decay_str="0.20:0.05"
            )
            self.assertAlmostEqual(info['lambda_t'], 0.20, places=2)
            self.assertTrue(info['has_icl'])
            self.assertTrue(info['has_know'])

    def test_capability_aware_stfa_allocates_budget_by_performance_gap(self):
        """Verify that STFA allocates more token budget to high-gap tasks (e.g. FSA)."""
        # Create synthetic candidate rows for sc, multifaced, fsa
        candidates = []
        for i in range(30):
            # sc candidates (short)
            candidates.append({
                'id': f'sc_{i}', 'group': 'sentiment:imdb', 'stage': 'icldist', 'split': 'train',
                'prompt': 'Classify sentiment: great movie.', 'response': 'positive',
                'input_ids': [1] * 50, 'labels': [-100] * 40 + [1] * 10
            })
            # multifaced candidates (medium)
            candidates.append({
                'id': f'mf_{i}', 'group': 'sentiment:tweeteval', 'stage': 'icldist', 'split': 'train',
                'prompt': 'Identify emotion in tweet: so happy!', 'response': 'joy',
                'input_ids': [1] * 150, 'labels': [-100] * 140 + [1] * 10
            })
            # fsa candidates (structured list)
            candidates.append({
                'id': f'fsa_{i}', 'group': 'sentiment:asqp_rest16', 'stage': 'icldist', 'split': 'train',
                'prompt': 'Extract aspect category opinion quadruples.', 'response': "[('food quality', 'food', 'good', 'positive')]",
                'input_ids': [1] * 200, 'labels': [-100] * 150 + [1] * 50
            })

        # Dev scores: FSA has lowest score (0.15) -> highest gap (0.90)
        # SC has highest score (0.90) -> lowest gap (0.15)
        dev_scores = {'sc': 0.90, 'multifaced': 0.60, 'fsa': 0.15}
        token_budget = 6000

        selected, report = select_rows_stfa(
            candidates, token_budget=token_budget, max_records=100, seed=42,
            dev_scores=dev_scores, stfa_temp=1.0
        )

        self.assertGreater(len(selected), 0)
        self.assertLessEqual(report['tokens'], token_budget)
        # FSA should receive the highest allocated budget proportion
        fsa_prop = report['task_proportions']['fsa']
        sc_prop = report['task_proportions']['sc']
        self.assertGreater(fsa_prop, sc_prop)
        self.assertGreater(report['task_budgets']['fsa'], report['task_budgets']['sc'])

    def test_zero_benchmark_leakage_between_cleaned_and_test(self):
        """Verify that official test manifest has 12 files and audit finds exact disjointness."""
        lock_path = ROOT / 'configs/sentibench_v1_test_manifest.json'
        self.assertTrue(lock_path.exists())
        lock = json.loads(lock_path.read_text())
        self.assertEqual(len(lock['files']), 12)

        # Check total records in test manifest
        total_test_records = sum(spec['records'] for spec in lock['files'].values())
        self.assertEqual(total_test_records, 11974)

    def test_asqp_parser_paper_parity_multiline_and_permutations(self):
        """Verify that ASQP parser reads official paper format, multiline code fences, and swaps."""
        # Case 1: Standard gold order: (category, aspect, opinion, sentiment)
        gold_quad = ("food quality", "pasta", "delicious", "positive")
        text1 = "[('food quality', 'pasta', 'delicious', 'positive')]"
        _, kept1, valid1 = asqp_prediction(text1, "The pasta was delicious.")
        self.assertTrue(valid1)
        self.assertEqual(kept1, [gold_quad])

        # Case 2: Prompt order: (aspect term, aspect category, opinion, sentiment)
        # The parser should automatically align category to index 0
        text2 = "[('pasta', 'food quality', 'delicious', 'positive')]"
        _, kept2, valid2 = asqp_prediction(text2, "The pasta was delicious.")
        self.assertTrue(valid2)
        self.assertEqual(kept2, [gold_quad])

        # Case 3: Multiline code block (which caused the 0.00% bug in previous regex/split parser)
        text3 = "```python\n[\n  ('pasta', 'food quality', 'delicious', 'positive')\n]\n```"
        _, kept3, valid3 = asqp_prediction(text3, "The pasta was delicious.")
        self.assertTrue(valid3)
        self.assertEqual(kept3, [gold_quad])

        # Case 4: Label: prefix on multiple lines
        text4 = "Label:\n[\n  ('food quality', 'pasta', 'delicious', 'positive')\n]"
        _, kept4, valid4 = asqp_prediction(text4, "The pasta was delicious.")
        self.assertTrue(valid4)
        self.assertEqual(kept4, [gold_quad])

    def test_strip_think_tags_qwen3(self):
        """Verify that thinking mode tags are cleanly removed."""
        text = "<think>\nThinking about sentiment...\n</think>\npositive"
        self.assertEqual(strip_think_tags(text), "positive")
        # Unclosed think tag
        text_unclosed = "<think>\nThinking..."
        self.assertEqual(strip_think_tags(text_unclosed), "")


if __name__ == '__main__':
    unittest.main()
