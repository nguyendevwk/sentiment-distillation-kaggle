"""Pure-Python invariant checks; no downloads, dependency install or model execution."""
import copy
from collections import Counter
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from distill_core import GroupWeights, digest, mapped_record, split_records


class CorpusTests(unittest.TestCase):
    def test_mapping_preserves_teacher_and_requires_explicit_source(self):
        raw = {'instruction': ' task \n', 'input': '', 'text': 'source',
               'answer': ' ["teacher"]\n ', 'cluster': 0}
        result = mapped_record(raw, ['instruction', 'text'], 'answer', 'cluster', 'text')
        self.assertEqual(result['response'], raw['answer'])
        self.assertEqual(result['prompt'], ' task \n\n\nsource')
        self.assertEqual(result['group'], '0')
        with self.assertRaises(ValueError):
            mapped_record(raw, ['instruction'], 'answer', 'cluster', 'input')

    def test_group_split_no_source_leakage_and_deterministic_pilot(self):
        rows = [{'id': f'{i}:{g}', 'source_key': digest(str(i)), 'group': g}
                for i in range(200) for g in ['a', 'b', 'c']]
        args = dict(seed=42, max_train=90, probe_per_group=5, holdout_per_group=5)
        first = split_records(copy.deepcopy(rows), **args)
        second = split_records(copy.deepcopy(rows), **args)
        self.assertEqual(first, second)
        sets = [{r['source_key'] for r in split} for split in first[:3]]
        for i, values in enumerate(sets):
            for other in sets[i + 1:]:
                self.assertFalse(values & other)
        self.assertEqual(len(first[0]), 90)

    def test_rare_group_mapping_comes_from_train(self):
        rows = [{'id': str(i), 'source_key': digest(i), 'group': f'g{i % 20}'} for i in range(500)]
        train, probe, holdout, mapping = split_records(rows, 7, max_groups=4, min_group_train=1)
        self.assertLessEqual(len(set(mapping.values())), 4)
        self.assertTrue(all(r['train_group'] in set(mapping.values()) for r in train + probe + holdout))


class WeightTests(unittest.TestCase):
    def make(self, mode='progress', **kw):
        return GroupWeights(['a', 'b', 'c'], {'a': 100, 'b': 10, 'c': 3}, mode=mode, **kw)

    def assert_bounded(self, controller):
        w = controller.weights
        self.assertTrue(all(math.isfinite(v) and 0.5 <= v <= 2 for v in w.values()))
        self.assertAlmostEqual(sum(w[g] * controller.counts[g] for g in w) / 113, 1.)

    def test_progress_rewards_learning_not_stubborn_high_loss(self):
        c = self.make(ema=0.)
        self.assertEqual(c.update({'a': 1., 'b': 8., 'c': 2.}), {'a': 1., 'b': 1., 'c': 1.})
        c.update({'a': 0.5, 'b': 8., 'c': 2.})
        self.assertGreater(c.weights['a'], c.weights['b'])
        self.assert_bounded(c)

    def test_no_progress_and_missing_probe_do_not_drop_groups(self):
        c = self.make()
        c.update({'a': 1., 'b': 2.})
        c.update({'a': 1.1, 'b': 2.2})
        self.assertTrue(all(abs(v - 1) < 1e-8 for v in c.weights.values()))
        self.assert_bounded(c)

    def test_balanced_bounded_and_uniform_immutable(self):
        b = self.make('balanced')
        self.assertGreater(b.weights['b'], b.weights['a'])
        self.assert_bounded(b)
        u = self.make('uniform')
        u.update({'a': 1., 'b': 2.})
        u.update({'a': 0.1, 'b': 3.})
        self.assertEqual(u.weights, {'a': 1., 'b': 1., 'c': 1.})

    def test_invalid_controls_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.make(temperature=0)
        with self.assertRaises(ValueError):
            self.make().update({'a': float('nan')})


if __name__ == '__main__':
    unittest.main()
