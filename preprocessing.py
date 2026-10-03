from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_pipeline():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_encoder, CATEGORICAL_COLUMNS),
    ])

    classifier = LogisticRegression(max_iter=1000)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", classifier),
    ])
    return model
