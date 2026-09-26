import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tests.fixtures import make_frame
from telco_churn.artifacts import ArtifactError, load_bundle, save_bundle
from telco_churn.config import TrainingConfig
from telco_churn.data import FEATURE_COLUMNS, validate_frame
from telco_churn.preprocessing import build_pipeline


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        frame = validate_frame(make_frame(20), labeled=True)
        self.X = frame[FEATURE_COLUMNS]
        self.model = build_pipeline('logistic',42).fit(self.X,frame.Churn)
        self.metadata = {'threshold':0.4, 'config':TrainingConfig().to_dict(), 'source_sha256':'a'*64}
        self.reports = {'split.json':{}, 'cv_results.json':[], 'validation.json':{}}

    def test_round_trip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'run'
            save_bundle(path,self.model,self.metadata,self.reports)
            model,metadata = load_bundle(path)
            np.testing.assert_allclose(model.predict_proba(self.X),self.model.predict_proba(self.X))
            self.assertEqual(metadata['threshold'],0.4)
            with self.assertRaises(FileExistsError):
                save_bundle(path,self.model,self.metadata,self.reports)

    def test_rejects_corrupt_incompatible_and_incomplete_before_unpickle(self):
        for mode in ('pickle','environment','source','schema','threshold','incomplete','report'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                path = Path(d)/'run'
                save_bundle(path,self.model,self.metadata,self.reports)
                meta_path = path/'metadata.json'
                meta = json.loads(meta_path.read_text())
                if mode == 'pickle': (path/'pipeline.pkl').write_bytes(b'broken')
                elif mode == 'incomplete': (path/'COMPLETE').unlink()
                elif mode == 'report': (path/'split.json').write_text('{} changed')
                elif mode == 'environment': meta['environment']['numpy'] = '0.0'
                elif mode == 'source': meta['code_sha256'] = '0'*64
                elif mode == 'schema': meta['schema']['features'] = []
                else: meta['threshold'] = float('nan')
                meta_path.write_text(json.dumps(meta))
                with patch('telco_churn.artifacts.pickle.load', side_effect=AssertionError('must not unpickle')):
                    with self.assertRaises(ArtifactError): load_bundle(path)

    def test_rejects_unsafe_report_names(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            save_bundle(Path(d)/'run',self.model,self.metadata,{'../elsewhere':{}})
