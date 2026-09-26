"""One pipeline is cloned during CV and saved intact for prediction."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, normalize_features


class SchemaNormalizer(TransformerMixin, BaseEstimator):
    """Stateless validation also runs when a saved pipeline is used directly."""
    def fit(self, X, y=None):
        normalize_features(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return normalize_features(X)


def build_pipeline(model_name: str, seed: int) -> Pipeline:
    factories = {
        'dummy': lambda: DummyClassifier(strategy='prior'),
        'logistic': lambda: LogisticRegression(solver='lbfgs', max_iter=2000, random_state=seed),
        'forest': lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
    }
    if model_name not in factories:
        raise ValueError(f'Unknown model: {model_name}.')
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scaler', StandardScaler())])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, NUMERIC_COLUMNS),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([('schema', SchemaNormalizer()), ('preprocess', preprocessing),
                     ('model', factories[model_name]())])
