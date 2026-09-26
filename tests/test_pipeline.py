import unittest

import numpy as np
from sklearn.base import clone
from tests.fixtures import make_frame
from telco_churn.data import FEATURE_COLUMNS, SchemaError, validate_frame
from telco_churn.preprocessing import build_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = validate_frame(make_frame(), labeled=True)
        self.X = self.frame[FEATURE_COLUMNS]
        self.y = self.frame.Churn

    def test_fit_statistics_exclude_heldout(self):
        model = build_pipeline('logistic',42).fit(self.X.iloc[:20], self.y.iloc[:20])
        imputer = model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['imputer']
        np.testing.assert_allclose(imputer.statistics_, [9.5,29.5,195])
        heldout = self.X.iloc[20:].copy()
        heldout['TotalCharges'] = 1e10
        model.predict_proba(heldout)
        np.testing.assert_allclose(imputer.statistics_, [9.5,29.5,195])

    def test_all_missing_numeric_retains_finite_transform(self):
        X = self.X.assign(TotalCharges=np.nan)
        model = build_pipeline('logistic',42).fit(X,self.y)
        transformed = model[:-1].transform(X)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.isfinite(model.predict_proba(self.X)).all())
        self.assertEqual(transformed.shape[1], model[:-1].transform(self.X).shape[1])
        numeric = model.named_steps['preprocess'].named_transformers_['numeric'].transform(
            X[['tenure','MonthlyCharges','TotalCharges']])
        self.assertEqual(numeric.shape[1],3)
        np.testing.assert_array_equal(numeric[:,2],np.zeros(len(X)))

    def test_allowed_unseen_category_and_column_order(self):
        for name in ('dummy','logistic','forest'):
            with self.subTest(name=name):
                pipeline = build_pipeline(name,42)
                if name == 'forest': pipeline.set_params(model__n_estimators=5)
                model = clone(pipeline).fit(self.X, self.y)
                np.testing.assert_allclose(model.predict_proba(self.X), model.predict_proba(self.X.iloc[:,::-1]))
                self.assertTrue(np.isfinite(model.predict_proba(self.X.assign(Contract='Two year'))).all())
                with self.assertRaises(SchemaError):
                    model.predict_proba(self.X.assign(Contract='invented'))
