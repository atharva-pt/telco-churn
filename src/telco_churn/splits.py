"""Reproducible customer-level partitions, bound to exact source bytes."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import file_sha256
from .config import TrainingConfig


def make_split_ids(frame: pd.DataFrame, config: TrainingConfig) -> dict[str, list[str]]:
    ordered = frame.sort_values('customerID')
    if set(ordered.Churn.unique()) != {0, 1} or ordered.customerID.duplicated().any():
        raise ValueError('Splitting requires unique IDs and both target classes.')
    try:
        development, test = train_test_split(ordered, test_size=config.test_fraction,
                                            stratify=ordered.Churn, random_state=config.seed)
        train, validation = train_test_split(
            development, test_size=config.validation_fraction / (1-config.test_fraction),
            stratify=development.Churn, random_state=config.seed)
    except ValueError as exc:
        raise ValueError('Insufficient class counts for the requested stratified splits.') from exc
    result = {}
    for name, part in [('train', train), ('validation', validation), ('test', test)]:
        if set(part.Churn.unique()) != {0, 1}:
            raise ValueError(f'{name}: insufficient class counts; both classes are required.')
        result[name] = sorted(part.customerID.tolist())
    return result


def select_partition(frame: pd.DataFrame, ids: dict[str, list[str]], name: str) -> pd.DataFrame:
    return frame.set_index('customerID', drop=False).loc[ids[name]].copy()


def create_manifest(data_path: Path, frame: pd.DataFrame, config: TrainingConfig) -> dict:
    return {'format_version': 1, 'source_sha256': file_sha256(data_path),
            'config': config.to_dict(), 'ids': make_split_ids(frame, config)}


def verify_manifest(manifest: dict, data_path: Path, frame: pd.DataFrame, config: TrainingConfig) -> None:
    if not isinstance(manifest, dict) or manifest.get('format_version') != 1:
        raise ValueError('Unsupported split manifest format.')
    if manifest.get('source_sha256') != file_sha256(data_path):
        raise ValueError('Source data fingerprint does not match split manifest.')
    if manifest.get('config') != config.to_dict():
        raise ValueError('Configuration does not match split manifest.')
    if manifest.get('ids') != make_split_ids(frame, config):
        raise ValueError('Split IDs are changed, overlapping, incomplete, or not reproducible.')
