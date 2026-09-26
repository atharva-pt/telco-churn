"""Batch inference consumes the saved pipeline; it never calls fit."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_bundle
from .data import FEATURE_COLUMNS, read_dataset


def predict_batch(data_path: Path, run_dir: Path, output_path: Path) -> pd.DataFrame:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Prediction output already exists: {output_path}')
    model,metadata = load_bundle(run_dir)
    frame = read_dataset(data_path,labeled=False)
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:,list(model.classes_).index(1)]
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('Model produced invalid probabilities.')
    result = pd.DataFrame({'customerID':frame.customerID.to_numpy(),
                           'churn_probability':probabilities,
                           'predicted_churn':(probabilities >= metadata['threshold']).astype(int),
                           'threshold':metadata['threshold'], 'run_id':metadata['run_id']})
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x',encoding='utf-8',newline='') as stream:
        result.to_csv(stream,index=False,float_format='%.17g')
    logging.info('Scored %d rows into %s.',len(result),output_path)
    return result
