"""Real invariants and CPU-only integration fixtures; never load a model."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from distill_core import digest, file_hash, read_records, write_json
from v1_core import (canonical_record, iter_objects, limited_candidates,
                     select_rows, source_text, split_name, steps_for_tokens)
from prepare_v1_pruned import jsonl, normalize, select
from run_v1_ab import build_request, commands


def fixture_rows():
    result = []
    for stage in ('knowdist', 'icldist'):
        for split in ('train', 'probe', 'holdout'):
            for g in ('analysis', 'rewrite'):
                for i in range(12):
                    size = 3 + i % 4
                    result.append({'id': f'{stage}:{split}:{g}:{i}', 'stage': stage, 'split': split,
                        'group': g, 'train_group': g, 'source_key': digest([split, i]),
                        'input_ids': [1] * size, 'labels': [-100] + [1] * (size - 1),
                        'prompt_tokens': 1, 'response_tokens': size - 1})
    return result


class NormalizationTests(unittest.TestCase):
    def test_failed_normalization_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'raw', Path(tmp) / 'normalized'
            root.mkdir()
            (root / 'analysis_basic.json').write_text(json.dumps([
                {'prompt': f'Analyze\n\nReview: {i}', 'llm_response': 'teacher', 'instr_type': 'analysis_basic'}
                for i in range(50)]))
            args = argparse.Namespace(input=str(root), output=str(out), revision='fixture', mapping_confirmed=True,
                                      benchmark_root=None, seed=42, dev_fraction=.2)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Missing train stage'):
                    normalize(args)
            self.assertFalse(out.exists())
            self.assertTrue(list(Path(tmp).glob('.normalized.failed-*/canonical.jsonl')))
            self.assertEqual(args.output, str(out))
            (root / 'sentiment_distillation_data_80k.json').write_text(json.dumps([
                {'instruction': f'Sentence: {i}\nOutput: ', 'output': 'positive'} for i in range(50)]))
            with contextlib.redirect_stdout(io.StringIO()):
                normalize(args)
            self.assertTrue((out / 'manifest.json').is_file())

    def test_stream_formats_bom_large_and_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.json'
            expected = [{'text': 'x' * 100000}, {'text': 'đúng'}]
            p.write_text('\ufeff' + json.dumps(expected))
            self.assertEqual([r for _, r in iter_objects(p)], expected)
            p.write_text('\n'.join(json.dumps(r) for r in expected))
            self.assertEqual([r for _, r in iter_objects(p)], expected)
            for bad in ('[{"a":1},]', '[{"a":1}', '[{"a":1}]xxx', '[{"a":1} {"b":2}]', '[3]'):
                p.write_text(bad)
                with self.assertRaises(ValueError):
                    list(iter_objects(p))

    def test_preservation_and_cross_perspective_source(self):
        raw = {'prompt': 'Analyze.\n\nReview: A mixed review. ', 'llm_response': ' Teacher\n',
               'instr_type': 'analysis_basic', 'dataset_name': 'amazon'}
        a = canonical_record(raw, 'analysis_basic.json', 1, 'sha')
        raw.update(prompt='Rewrite.\n\nText: A mixed review. ', instr_type='rewrite_basic')
        b = canonical_record(raw, 'rewrite_basic.json', 1, 'sha2')
        self.assertEqual(a['source_key'], b['source_key'])
        self.assertEqual(a['response'], ' Teacher\n')
        self.assertEqual(a['prompt'], 'Analyze.\n\nReview: A mixed review. ')
        icl = canonical_record({'instruction': 'Sentence: demo\nOutput: negative\n\nSentence: A mixed review.\nOutput: ',
                                'input': '', 'output': 'positive '}, 'sentiment_distillation_data_80k.json', 1, 'sha3')
        self.assertEqual(icl['source_key'], a['source_key'])
        self.assertEqual(icl['response'], 'positive ')
        for seed in (42, 52):
            self.assertEqual(split_name(a['source_key'], seed, .2), split_name(icl['source_key'], seed, .2))

    def test_unknown_source_is_not_silently_prompt_hash(self):
        with self.assertRaisesRegex(ValueError, 'source_unresolved'):
            source_text('A task with unknown framing', 'knowdist')
        with self.assertRaisesRegex(ValueError, 'unknown_schema'):
            canonical_record({'text': 'unknown'}, 'a.json', 1, 'sha')

    def test_paired_inputs_keep_both_fields_and_ignore_demonstrations(self):
        for first, second in [('Premise', 'Hypothesis'), ('Sentence1', 'Sentence2')]:
            prompt = f'{first}: demo\n{second}: example\nOutput: x\n\n{first}: actual\n{second}: pair\nOutput: '
            self.assertEqual(source_text(prompt, 'icldist'), f'actual\n{second}: pair')

    def test_normalize_dedup_and_benchmark_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'raw', Path(tmp) / 'normalized'
            root.mkdir()
            know = [{'prompt': f'Analyze\n\nReview: source {i}', 'llm_response': 'unchanged',
                     'instr_type': 'analysis_basic'} for i in range(100)]
            (root / 'analysis_basic.json').write_text(json.dumps(know + [know[0]]))
            icl = [{'instruction': f'Sentence: source {i}\nOutput: ', 'input': '', 'output': 'positive'} for i in range(100)]
            (root / 'sentiment_distillation_data_80k.json').write_text(json.dumps(icl))
            benchmark = Path(tmp) / 'bench' / 'sc' / 'sst2'
            benchmark.mkdir(parents=True)
            (benchmark / 'test.json').write_text(json.dumps([{'sentence': 'source 1'}]))
            args = argparse.Namespace(input=str(root), output=str(out), revision='synthetic-test',
                    mapping_confirmed=True, benchmark_root=str(Path(tmp) / 'bench'), seed=42, dev_fraction=.2)
            with contextlib.redirect_stdout(io.StringIO()):
                normalize(args)
            rows = [r for _, r in read_records(out / 'canonical.jsonl')]
            self.assertEqual(len(rows), 198)
            self.assertEqual(len({r['id'] for r in rows}), len(rows))
            groups = {}
            for r in rows:
                groups.setdefault(r['source_key'], set()).add(r['split'])
            self.assertTrue(all(len(s) == 1 for s in groups.values()))
            exclusions = [r for _, r in read_records(out / 'exclusions.jsonl')]
            self.assertTrue(any('benchmark' in r['reason'] for r in exclusions))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [r for r in fixture_rows() if r['stage'] == 'knowdist' and r['split'] == 'train']

    def test_selection_strict_budget_determinism_and_coverage(self):
        for method in ('coverage', 'random', 'student'):
            scores = {r['id']: .5 + int(r['id'].split(':')[-1]) / 100 for r in self.rows}
            a, report = select_rows(copy.deepcopy(self.rows), method, 40, 20, 42, scores)
            b, _ = select_rows(list(reversed(copy.deepcopy(self.rows))), method, 40, 20, 42, scores)
            self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
            self.assertEqual(report['tokens'], sum(len(r['input_ids']) for r in a))
            self.assertLessEqual(report['tokens'], 40)
            if method != 'random':
                self.assertEqual(set(report['groups']), {'analysis', 'rewrite'})
            self.assertEqual(len(a), len({r['id'] for r in a}))

    def test_missing_nonfinite_score_and_dev_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing score'):
            select_rows(self.rows, 'student', 100, 20, 42, {})
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            select_rows(self.rows, 'student', 100, 20, 42, {r['id']: float('nan') for r in self.rows})
        dev = copy.deepcopy(self.rows)
        dev[0]['split'] = 'holdout'
        with self.assertRaisesRegex(ValueError, 'train only'):
            select_rows(dev, 'coverage', 100, 20, 42)

    def test_candidate_reservoir_independent_of_input_order(self):
        a = limited_candidates(fixture_rows(), 42, 6, {'knowdist': 8, 'icldist': 7})
        b = limited_candidates(list(reversed(fixture_rows())), 42, 6, {'knowdist': 8, 'icldist': 7})
        self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
        self.assertEqual(sum(r['stage'] == 'knowdist' and r['split'] == 'train' for r in a), 8)

    def test_budget_simulation_matches_training_batch_order(self):
        rows = self.rows[:5]
        steps, planned = steps_for_tokens(rows, 107, 2, 3, 42)
        # Independent literal trainer stream simulation, with batches crossing epochs.
        epoch, offset, order, total = 0, 0, None, 0
        for _ in range(steps):
            for _ in range(3):
                batch = []
                while len(batch) < 2:
                    if order is None:
                        order = list(range(len(rows)))
                        random.Random(42 + epoch).shuffle(order)
                    take = min(2 - len(batch), len(order) - offset)
                    batch.extend(rows[i] for i in order[offset:offset + take])
                    offset += take
                    if offset == len(order):
                        epoch, offset, order = epoch + 1, 0, None
                total += sum(len(r['input_ids']) for r in batch)
        self.assertEqual(total, planned)
        self.assertGreaterEqual(total, 107)
        self.assertLess(total - 107, 6 * max(len(r['input_ids']) for r in rows))

    def test_ab_preparation_shared_icl_dev_and_stage_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / 'pool'
            (pool / 'tokenizer').mkdir(parents=True)
            (pool / 'tokenizer' / 'tokenizer_config.json').write_text('{}')
            jsonl(pool / 'pool.jsonl', fixture_rows())
            manifest = {'pool_sha256': file_hash(pool / 'pool.jsonl'), 'model': 'fixture-model',
                        'model_revision': 'fixture-sha', 'corpus_revision': 'fixture-corpus',
                        'tokenizer_hashes': {'tokenizer_config.json': file_hash(pool / 'tokenizer/tokenizer_config.json')},
                        'tokenizer_template_sha256': 'fixture-template'}
            write_json(pool / 'manifest.json', manifest)
            scores_dir = root / 'scores'
            scores_dir.mkdir()
            jsonl(scores_dir / 'scores.jsonl', [{'id': r['id'], 'ifd_proxy': .8} for r in self.rows])
            write_json(scores_dir / 'identity.json', {k: manifest[k] for k in ('pool_sha256', 'model', 'model_revision')})
            write_json(scores_dir / 'summary.json', {'completed': True,
                'identity_sha256': file_hash(scores_dir / 'identity.json'), 'scores_sha256': file_hash(scores_dir / 'scores.jsonl')})
            for name, method in [('A', 'student'), ('B', 'coverage'), ('R', 'random')]:
                args = argparse.Namespace(pool=str(pool), output=str(root / name), method=method,
                    scores=str(scores_dir), know_tokens=40, icl_tokens=40, know_records=20, icl_records=20,
                    seed=42, min_ifd=0., max_ifd=1., allow_missing_groups=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    select(args)
            for stage, split in [('icldist', 'train'), ('knowdist', 'probe'), ('knowdist', 'holdout'), ('icldist', 'holdout')]:
                self.assertEqual((root / 'A' / stage / f'{split}.jsonl').read_bytes(),
                                 (root / 'B' / stage / f'{split}.jsonl').read_bytes())
            config = json.loads((Path(__file__).parent.parent / 'configs/v1_ab.json').read_text())
            request = build_request(root / 'A', config, 'A', 42, pilot_steps=3)
            cmds = commands(request, root / 'run')
            self.assertNotIn('--init_adapter', cmds[0][1])
            self.assertIn('--init_adapter', cmds[1][1])
            self.assertEqual(request['stages']['knowdist']['steps'], 3)
            # A must fail on mismatched score identity; prepared B remains usable.
            write_json(scores_dir / 'identity.json', {'pool_sha256': 'wrong'})
            args.output, args.method = str(root / 'badA'), 'student'
            with self.assertRaises(ValueError):
                select(args)
            self.assertTrue((root / 'B/icldist/manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
