import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tests.fixtures import make_frame
from tests.test_training import prepare_run
from telco_churn.predict import predict_batch
from telco_churn.train import train_run
from telco_churn.preprocessing import SchemaNormalizer


class PredictionTests(unittest.TestCase):
    def test_prediction_preserves_rows_threshold_and_order_without_fitting(self):
        with tempfile.TemporaryDirectory() as d:
            data,config,split,run,_=prepare_run(d)
            train_run(data,config,split,run)
            raw=make_frame(7).drop(columns='Churn').iloc[::-1]
            batch=Path(d)/'batch.csv'; output=Path(d)/'scores.csv'
            raw.to_csv(batch,index=False)
            with patch.object(SchemaNormalizer,'fit',side_effect=AssertionError('prediction must not fit')):
                predictions=predict_batch(batch,run,output)
            self.assertEqual(predictions.customerID.tolist(),raw.customerID.tolist())
            self.assertTrue(predictions.churn_probability.between(0,1).all())
            np.testing.assert_array_equal(predictions.predicted_churn,
                (predictions.churn_probability >= predictions.threshold).astype(int))
            raw.iloc[:,::-1].to_csv(batch,index=False)
            second=predict_batch(batch,run,Path(d)/'reordered.csv')
            np.testing.assert_allclose(predictions.churn_probability,second.churn_probability)
            with self.assertRaises(FileExistsError): predict_batch(batch,run,output)
            make_frame(1).drop(columns='Churn').to_csv(batch,index=False)
            self.assertEqual(len(predict_batch(batch,run,Path(d)/'single.csv')),1)

    def test_labeled_and_empty_batches_fail(self):
        with tempfile.TemporaryDirectory() as d:
            data,config,split,run,_=prepare_run(d)
            train_run(data,config,split,run)
            with self.assertRaises(ValueError): predict_batch(data,run,Path(d)/'bad.csv')
            empty=Path(d)/'empty.csv'
            make_frame().drop(columns='Churn').iloc[:0].to_csv(empty,index=False)
            with self.assertRaises(ValueError): predict_batch(empty,run,Path(d)/'empty-out.csv')
