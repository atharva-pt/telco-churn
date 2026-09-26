import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tests.fixtures import make_frame
from telco_churn.artifacts import load_bundle, write_json_new
from telco_churn.config import TrainingConfig
from telco_churn.data import read_dataset
from telco_churn.metrics import choose_threshold, threshold_metrics
from telco_churn.preprocessing import SchemaNormalizer
from telco_churn.splits import create_manifest
from telco_churn.train import train_run


def prepare_run(directory, n=100, config=None):
    directory = Path(directory)
    data = directory/'data.csv'
    make_frame(n).to_csv(data,index=False)
    config = config or TrainingConfig(cv_folds=2, logistic_c=(1,), forest_trees=5,
                                       forest_depths=(3,), forest_min_leaf=(1,))
    manifest = create_manifest(data,read_dataset(data,labeled=True),config)
    manifest_path = directory/'split.json'
    write_json_new(manifest_path,manifest)
    return data,config,manifest_path,directory/'run',manifest


class TrainingTests(unittest.TestCase):
    def test_threshold_policies_and_equality(self):
        y=np.array([0,1]); p=np.array([0.2,0.8])
        self.assertEqual(choose_threshold(y,p,'max_f1'),0.8)
        self.assertEqual(choose_threshold(y,p,'fixed_0_5'),0.5)
        self.assertEqual(threshold_metrics(y,p,0.8)['recall'],1)
        self.assertEqual(choose_threshold(np.array([1,0,0,1]),np.array([0.1,0.2,0.3,0.4]),'max_f1'),0.4)
        for bad in [np.array([np.nan,0.5]),np.array([-1,0.5]),np.array([0.2,2])]:
            with self.assertRaises(ValueError): choose_threshold(y,bad,'max_f1')

    def test_only_training_rows_enter_fit_and_model_is_not_refit(self):
        with tempfile.TemporaryDirectory() as d:
            data,config,split,run,manifest=prepare_run(d)
            fits=[]
            original=SchemaNormalizer.fit
            def observe(estimator,X,y=None):
                fits.append(set(X.index))
                return original(estimator,X,y)
            with patch.object(SchemaNormalizer,'fit',new=observe):
                train_run(data,config,split,run)
            training_ids=set(manifest['ids']['train'])
            self.assertEqual(len(fits),7)  # 3 candidates * 2 folds + one refit.
            self.assertTrue(all(ids <= training_ids for ids in fits))
            self.assertEqual(fits[-1],training_ids)
            model,meta=load_bundle(run)
            self.assertFalse((run/'evaluation').exists())
            results=json.loads((run/'cv_results.json').read_text())
            self.assertEqual(len(results),3)
            self.assertEqual(len(results[0]['fold_scores']),2)
            frame=read_dataset(data,labeled=True).set_index('customerID')
            median=frame.loc[manifest['ids']['train'],'TotalCharges'].median()
            stored=model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['imputer'].statistics_[2]
            self.assertEqual(stored,median)
            self.assertGreaterEqual(meta['threshold'],0)

    def test_insufficient_cv_class_counts_fail_before_fit(self):
        with tempfile.TemporaryDirectory() as d:
            args=prepare_run(d,n=20,config=TrainingConfig(cv_folds=10))
            with self.assertRaisesRegex(ValueError,'class counts'):
                train_run(*args[:4])
