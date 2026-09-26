"""The input contract, shared by training and prediction.

Normalization is deterministic. Imputation belongs in the fitted pipeline.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
]
NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]
CATEGORIES = {
    'gender': ['Female', 'Male'], 'SeniorCitizen': ['0', '1'],
    'Partner': ['No', 'Yes'], 'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'], 'MultipleLines': ['No', 'Yes', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    **{c: ['No', 'Yes', 'No internet service'] for c in
       ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']},
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
}


class SchemaError(ValueError):
    """Input cannot be safely interpreted under this model's schema."""


def _check_columns(frame: pd.DataFrame, expected: list[str]) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SchemaError('Input must be a nonempty DataFrame.')
    if frame.columns.duplicated().any():
        raise SchemaError('Duplicate column headers are not allowed.')
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise SchemaError(f'Column mismatch: missing={missing}, unexpected={extra}.')


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate exactly 19 features, without learning any statistics."""
    _check_columns(frame, FEATURE_COLUMNS)
    result = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        text = result[column].astype('string').str.strip()
        missing = text.isna() | text.eq('').fillna(False)
        try:
            numeric = pd.to_numeric(text.mask(missing), errors='raise').astype(float)
        except (ValueError, TypeError) as exc:
            raise SchemaError(f'{column}: invalid numeric values.') from exc
        invalid = (~missing & ~np.isfinite(numeric)) | (numeric < 0)
        if column == 'tenure':
            invalid |= numeric.notna() & (numeric % 1 != 0)
        if invalid.any():
            raise SchemaError(f'{column}: {int(invalid.sum())} invalid numeric values.')
        result[column] = numeric
    for column in CATEGORICAL_COLUMNS:
        text = result[column].astype('string').str.strip()
        invalid = ~text.isin(CATEGORIES[column])
        if invalid.any():
            raise SchemaError(f'{column}: {int(invalid.sum())} missing or invalid categories.')
        result[column] = text.astype(object)
    return result


def validate_frame(frame: pd.DataFrame, *, labeled: bool) -> pd.DataFrame:
    """Validate raw records. Labeled input maps Yes/No to integer 1/0."""
    expected = ['customerID', *FEATURE_COLUMNS, *(['Churn'] if labeled else [])]
    _check_columns(frame, expected)
    ids = frame.customerID.astype('string').str.strip()
    if ids.isna().any() or ids.eq('').any() or ids.duplicated().any():
        raise SchemaError('customerID: missing or duplicate IDs.')
    result = normalize_features(frame[FEATURE_COLUMNS])
    result.insert(0, 'customerID', ids.astype(str))
    if labeled:
        target = frame.Churn.astype('string').str.strip()
        if not target.isin(['No', 'Yes']).all():
            raise SchemaError('Churn: expected nonmissing Yes/No labels.')
        result['Churn'] = target.map({'No': 0, 'Yes': 1}).astype(int)
    return result


def read_dataset(path: Path, *, labeled: bool) -> pd.DataFrame:
    """Parse without silently renaming duplicate headers or coercing IDs."""
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header or len(set(header)) != len(header):
                raise SchemaError('CSV has empty or duplicate headers.')
            rows = []
            for number, row in enumerate(reader, 2):
                if len(row) != len(header):
                    raise SchemaError(f'CSV row {number}: incorrect field count.')
                rows.append(row)
    except (csv.Error, UnicodeError) as exc:
        raise SchemaError('CSV is malformed or is not UTF-8.') from exc
    return validate_frame(pd.DataFrame(rows, columns=header), labeled=labeled)
