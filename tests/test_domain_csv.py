import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.prepare_domain_csv import materialize


class DomainCsvTests(unittest.TestCase):
    def test_identity_order_and_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'metadata.csv';selection=p/'selection.json'
            source.write_text('id,video path\na,videos/a.mp4\nb,videos/b.mp4\nc,videos/c.mp4\n')
            plan={'metadata_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'splits':{'train_single_2048':['b','a'],'train_mixed_2048':['a'], 'eval_street_128':['c']}}
            selection.write_text(json.dumps(plan))
            first=materialize(source,selection,p/'out')
            self.assertEqual(first,materialize(source,selection,p/'out'))
            with (p/'out/train_single_2048.csv').open(newline='') as f:
                self.assertEqual([r['id'] for r in csv.DictReader(f)],['b','a'])
            plan['splits']['eval_street_128']=['a'];selection.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'leakage'):materialize(source,selection,p/'out')
            plan['splits']['eval_street_128']=['c'];plan['metadata_sha256']='wrong';selection.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'hash'):materialize(source,selection,p/'out')


if __name__=='__main__':unittest.main()
