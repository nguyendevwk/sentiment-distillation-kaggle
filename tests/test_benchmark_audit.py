"""Test isolation/provenance checks, no model operations."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_benchmark_splits import audit, validate_final_test_request


class BenchmarkAuditTests(unittest.TestCase):
    def test_detects_text_overlap_without_using_labels_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'sc' / 'fixture'
            path.mkdir(parents=True)
            test = [{'sentence': ' SAME   text ', 'label': 'positive'}, {'sentence': 'only test', 'label': 'negative'}]
            (path / 'test.json').write_text(json.dumps(test))
            before = (path / 'test.json').read_bytes()
            (path / 'train.json').write_text(json.dumps([{'sentence': 'same text', 'label': 'different label'},
                                                        {'sentence': 'different text', 'label': 'positive'}]))
            (path / 'val.json').write_text(json.dumps([{'sentence': 'same text'}, {'sentence': 'development only'}]))
            manifest = {'repository': 'fixture', 'revision': 'fixture-sha', 'files': {
                'sc/fixture': {'records': 2, 'sha256': hashlib.sha256(before).hexdigest()}}}
            report = audit(root, manifest, pilot_samples=2)
            self.assertTrue(report['test_integrity_passed'])
            self.assertEqual(report['development_overlap_rows'], 2)
            self.assertEqual(report['datasets']['sc/fixture']['pilot_val']['exact_test_overlap_rows'], 1)
            self.assertEqual(before, (path / 'test.json').read_bytes())
            (path / 'test.json').write_text(json.dumps(test[:1]))
            self.assertFalse(audit(root, manifest)['test_integrity_passed'])

    def test_final_test_request_cannot_be_a_subset(self):
        validate_final_test_request(0, 1.)
        for count, proportion in [(64, 1.), (0, .5), (-1, 1.)]:
            with self.assertRaises(ValueError):
                validate_final_test_request(count, proportion)


if __name__ == '__main__':
    unittest.main()
