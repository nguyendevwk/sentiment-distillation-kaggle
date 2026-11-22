"""Launcher/evaluation provenance checks without torch imports or model execution."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import notebook_runtime
from run_kaggle_pilot import collect_report, eval_command, once, pipeline


class PilotTests(unittest.TestCase):
    def test_dry_run_does_not_write_or_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'not-created'
            args = argparse.Namespace(steps=50, eval_samples=64, max_length=2048,
                max_seconds=10800, work=str(path), dry_run=True, plan='B', model='fixture-model')
            with patch('run_kaggle_pilot.execute', side_effect=AssertionError('must not run')):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    pipeline(args)
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(output.getvalue())['eval_split'], 'val')

    def test_evaluation_uses_validation_not_test_and_pins_revision(self):
        args = argparse.Namespace(seed=42, eval_samples=64, max_length=2048)
        cmd = eval_command('fixture-model', 'immutable-sha', Path('/tmp/fixture-adapter'),
                           'sc/sst2', Path('/tmp/fixture-eval/before/sst2'), args)
        self.assertEqual(cmd[cmd.index('--eval_split') + 1], 'val')
        self.assertEqual(cmd[cmd.index('--demo_split') + 1], 'train')
        self.assertEqual(cmd[cmd.index('--model_revision') + 1], 'immutable-sha')
        self.assertEqual(cmd[cmd.index('--eval_batch_size') + 1], '1')

    def test_resume_receipt_rejects_changed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'state').mkdir()
            (work / 'logs').mkdir()
            output = work / 'output.json'
            def run_fixture(cmd, log):
                output.write_text('fixture')
            with patch('run_kaggle_pilot.execute', side_effect=run_fixture) as execute:
                once(work, 'fixture', ['fixture-command'], [output])
                once(work, 'fixture', ['fixture-command'], [output])
                self.assertEqual(execute.call_count, 1)
                output.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    once(work, 'fixture', ['fixture-command'], [output])

    def test_report_preserves_dev_metrics_and_timing_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            result = work / 'evaluation/before/sc_sst2/result.txt'
            result.parent.mkdir(parents=True)
            metrics = {'f1': .75, 'acc': .8, 'metadata': {'inference_measurement': {
                'amortized_ms_per_example': 12., 'peak_allocated_bytes': 1024**3}}}
            result.write_text(json.dumps(['fixture', metrics]) + '\n')
            args = argparse.Namespace(plan='B', seed=42, steps=50, eval_samples=64)
            report = collect_report(work, args, 10.)
            self.assertEqual(report['eval_split'], 'val')
            self.assertEqual(report['evaluations']['before']['sc_sst2']['f1'], .75)
            self.assertIn('75.00', (work / 'report-B.md').read_text())


class RuntimeProvenanceTests(unittest.TestCase):
    def test_random_dev_subset_is_stable_and_does_not_change_demo_rng(self):
        import random
        raw = {'test': [{'sentence': str(i)} for i in range(100)], 'demo': [1, 2, 3, 4]}
        args = argparse.Namespace(data_prop=1., max_samples=10, sample_strategy='random', seed=42, k_shot=4)
        state = random.getstate()
        a = notebook_runtime.select_examples({k: list(v) for k, v in raw.items()}, args)
        b = notebook_runtime.select_examples({k: list(v) for k, v in raw.items()}, args)
        self.assertEqual(a, b)
        self.assertEqual(state, random.getstate())
        self.assertNotEqual(a['test'], raw['test'][:10])

    def test_changed_adapter_invalidates_prediction_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            weights = path / 'adapter_model.safetensors'
            weights.write_bytes(b'synthetic-weights-one')
            args = argparse.Namespace(model_name_or_path='fixture', k_shot=4, max_seq_length=100,
                max_samples=10, data_prop=1., demo_split='train', resolved_dtype='float32',
                load_in_4bit=False, max_new_tokens=10, eval_batch_size=1, eval_split='val',
                sample_strategy='random', model_revision='sha', adapter=str(path), measure_inference=True,
                subname='fixture', dataset='sc/sst2', seed=42, resolved_model_revision='sha')
            with patch('importlib.metadata.version', return_value='fixture-version'):
                old = notebook_runtime.run_metadata(args, {'test': [], 'demo': []}, 0)
                cache = notebook_runtime.PredictionCache(str(path / 'cache.json'), old)
                cache.flush()
                weights.write_bytes(b'synthetic-weights-two')
                new = notebook_runtime.run_metadata(args, {'test': [], 'demo': []}, 0)
            self.assertNotEqual(old['adapter_sha256'], new['adapter_sha256'])
            with self.assertRaisesRegex(ValueError, 'differ'):
                notebook_runtime.PredictionCache(str(path / 'cache.json'), new)


if __name__ == '__main__':
    unittest.main()
