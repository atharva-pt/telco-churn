import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_training import prepare_run
from telco_churn.evaluate import evaluate_run
from telco_churn.preprocessing import SchemaNormalizer
from telco_churn.train import train_run


class EvaluationTests(unittest.TestCase):
    def test_evaluation_uses_frozen_holdout_and_refuses_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            data,config,split,run,manifest=prepare_run(d)
            train_run(data,config,split,run)
            with patch.object(SchemaNormalizer,'fit',side_effect=AssertionError('evaluation must not fit')):
                metrics=evaluate_run(data,run)
            self.assertEqual(metrics['sample_count'],20)
            self.assertEqual(sum(sum(row) for row in metrics['confusion_matrix']),20)
            self.assertTrue((run/'evaluation'/'COMPLETE').is_file())
            self.assertEqual(len(list((run/'evaluation').glob('*.png'))),3)
            with self.assertRaises(FileExistsError): evaluate_run(data,run)

    def test_changed_source_fails_before_evaluation_output(self):
        with tempfile.TemporaryDirectory() as d:
            data,config,split,run,_=prepare_run(d)
            train_run(data,config,split,run)
            data.write_text(data.read_text().replace('100,No','101,No',1))
            with self.assertRaises(ValueError): evaluate_run(data,run)
            self.assertFalse((run/'evaluation').exists())
