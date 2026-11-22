"""Full-test isolation, resume and structured-output checks without model imports."""
import argparse
import ast
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from evaluation_core import (prepare_clean_benchmark, sha, stable_digest,
                             inspect_trained_adapter, verify_result)
from notebook_runtime import PredictionCache, select_examples
from run_kaggle_evaluation import collect, result_paths
from parse_utils.structured_literal import (literal_tuples, asqp_prediction,
                                            ssa_prediction, atsa_prediction, acsa_prediction)


class FullEvaluationTests(unittest.TestCase):
    def test_report_refuses_avg12_until_all_datasets_and_seeds_complete(self):
        lock = json.loads((ROOT / 'configs/sentibench_v1_test_manifest.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            dataset = next(iter(lock['files']))
            directory, result, receipt = result_paths(tmp, 'before', dataset, 42)
            directory.mkdir(parents=True)
            metadata = {'dataset': dataset, 'seed': 42, 'num_samples': lock['files'][dataset]['records'],
                'settings': {'eval_split': 'test', 'max_samples': 0, 'data_prop': 1},
                'data_sha256': 'fixture', 'demo_manifest_sha256': 'fixture'}
            result.write_text(json.dumps(['fixture', {'f1': .5, 'acc': None, 'metadata': metadata}]))
            receipt.write_text(json.dumps({'result_sha256': sha(result)}))
            report = collect(tmp, {'before': {}}, lock, [42, 52, 62])
            self.assertIsNone(report['models']['before']['avg12_f1_percent'])
            self.assertFalse(report['models']['before']['complete'])
            self.assertEqual(report['models']['before']['datasets'][dataset]['seeds_completed'], 1)

    def test_clean_only_development_side_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / 'input', Path(tmp) / 'clean'
            dataset = root / 'sc/fixture'
            dataset.mkdir(parents=True)
            (dataset / 'test.json').write_text('[{"sentence":"held out","label":"x"}]')
            (dataset / 'train.json').write_text(json.dumps([
                {'sentence': 'HELD  OUT', 'label': 'different'}, {'sentence': 'development'}]))
            (dataset / 'val.json').write_text('[{"sentence":"validation"}]')
            lock = {'repository': 'fixture', 'revision': '1', 'files': {'sc/fixture': {
                'records': 1, 'sha256': sha(dataset / 'test.json')}}}
            report = prepare_clean_benchmark(root, output, lock)
            self.assertTrue(report['audit']['exact_benchmark_splits_disjoint'])
            self.assertEqual((dataset / 'test.json').read_bytes(), (output / 'sc/fixture/test.json').read_bytes())
            kept = json.loads((output / 'sc/fixture/train.json').read_text())
            self.assertEqual([r['sentence'] for r in kept], ['development'])
            self.assertTrue(kept[0]['_demo_id'])
            self.assertEqual(report, prepare_clean_benchmark(root, output, lock))
            (output / 'sc/fixture/train.json').write_text('[]')
            with self.assertRaises(ValueError):
                prepare_clean_benchmark(root, output, lock)

    def test_prediction_identity_handles_duplicate_prompts_and_resumes_timing(self):
        raw = {'test': [{'prompts': 'same'}, {'prompts': 'same'}], 'demo': []}
        select_examples(raw, argparse.Namespace(data_prop=1, max_samples=0, k_shot=0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.json'
            cache = PredictionCache(path, {'fixture': True})
            cache.put(raw['test'][0], 'negative')
            cache.measure(.5, 1, allocated=10)
            cache.flush()
            resumed = PredictionCache(path, {'fixture': True})
            self.assertIsNone(resumed.get(raw['test'][1]))
            resumed.put(raw['test'][1], 'positive')
            resumed.measure(.25, 1, allocated=12)
            measurement = resumed.measurement_summary('forward', 1)
            self.assertEqual(measurement['measured_examples'], 2)
            self.assertEqual(measurement['amortized_ms_per_example'], 375)
            self.assertEqual(measurement['peak_allocated_bytes'], 12)

    def test_partial_validation_results_cannot_count_as_full_test(self):
        lock = {'files': {'sc/fixture': {'records': 2}}}
        metadata = {'dataset': 'sc/fixture', 'seed': 42, 'num_samples': 2,
                    'settings': {'eval_split': 'test', 'max_samples': 0, 'data_prop': 1}}
        record = ['fixture', {'metadata': metadata}]
        verify_result(record, 'sc/fixture', lock, 42)
        metadata['num_samples'] = 1
        with self.assertRaises(ValueError):
            verify_result(record, 'sc/fixture', lock, 42)
        metadata['num_samples'] = 2
        metadata['settings']['eval_split'] = 'val'
        with self.assertRaises(ValueError):
            verify_result(record, 'sc/fixture', lock, 42)

    def test_audit_reconstructs_consumed_rows_and_detects_embedded_test_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            run = work / 'runs/fixture'
            run.mkdir(parents=True)
            (run / 'request.json').write_text('{"plan":"B"}')
            (run / 'completed.json').write_text(json.dumps({'request_sha256': sha(run / 'request.json')}))
            for stage in ('knowdist', 'icldist'):
                prepared, directory = work / 'prepared/B' / stage, run / stage
                prepared.mkdir(parents=True)
                (directory / 'adapter').mkdir(parents=True)
                candidates = [{'id': str(i), 'source_key': stable_digest('clean source'),
                    'prompt': 'Sentence: held out\nLabel: positive\n\nSentence: clean source\nLabel:',
                    'input_ids': [1] * (i + 1)} for i in range(3)]
                (prepared / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in candidates))
                (prepared / 'manifest.json').write_text(json.dumps({'model': 'fixture', 'model_revision': 'r1',
                    'split_hashes': {'train': sha(prepared / 'train.jsonl')}}))
                order = list(range(3))
                random.Random(42).shuffle(order)
                (directory / 'identity.json').write_text(json.dumps({'config': {
                    'batch_size': 1, 'grad_accum': 2, 'seed': 42},
                    'manifest_sha256': sha(prepared / 'manifest.json')}))
                (directory / 'summary.json').write_text(json.dumps({'completed': True, 'state': {
                    'step': 1, 'optimizer_updates': 1,
                    'train_tokens': sum(len(candidates[i]['input_ids']) for i in order[:2])}}))
                (directory / 'adapter/adapter_config.json').write_text('{}')
                (directory / 'artifact-integrity.json').write_text(json.dumps({
                    name: sha(directory / name) for name in ('identity.json', 'summary.json', 'adapter/adapter_config.json')}))
            report = inspect_trained_adapter(work, run, {'held out'})
            self.assertTrue(report['exact_checked_overlap_found'])
            self.assertEqual(len(report['confirmed_overlap_records']), 4)
            self.assertEqual(report['confirmed_overlap_records'][0]['reason'], ['marked_prompt_or_demo_input'])
            self.assertEqual(report['stages']['knowdist']['presentations'], 2)

    def test_safe_literal_parser_retains_all_rows_and_rejects_executable_text(self):
        quads = [('food quality', 'food', f'good{i}', 'positive') for i in range(5)]
        text = '```python\n' + json.dumps(quads, indent=2) + '\n```'
        self.assertEqual(asqp_prediction(text, 'food')[1], quads)
        self.assertEqual(literal_tuples("__import__('os').getcwd()", 4)[1], [])
        self.assertEqual(literal_tuples("[('unclosed'", 4)[1], [])

    def test_all_official_fsa_gold_labels_roundtrip(self):
        # Execute only the author's pure label-conversion method, with no torch imports.
        tree = ast.parse((ROOT / 'fsa_kaggle.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FineTuneDataset')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'make_label_seq')
        namespace = {'ifnt0': lambda v: 0 if v is None else v}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<gold-label-method>', 'exec'), namespace)
        parsers = {'asqp_rest16': lambda t, s: asqp_prediction(t, s)[1],
                   'opener': lambda t, s: ssa_prediction(t, s)[1],
                   'atsa_rest16': atsa_prediction, 'acsa_rest16': lambda t, s: acsa_prediction(t)}
        for name, parser in parsers.items():
            rows = json.loads((ROOT / 'datasets/absa' / name / 'test.json').read_text(encoding='utf-8-sig'))
            for index, row in enumerate(rows):
                namespace['make_label_seq'](None, 'absa/' + name, row)
                expected = [tuple(v) for v in row['label_seq'] if tuple(v) != ('NULL',) * 4]
                self.assertEqual(set(parser(repr(row['label_seq']), row['sentence'])), set(expected),
                                 msg=f'{name}/{index}')


if __name__ == '__main__':
    unittest.main()
