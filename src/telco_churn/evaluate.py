"""One explicit evaluation of a frozen run on its reserved test partition."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from .artifacts import load_bundle, write_json_new
from .config import TrainingConfig
from .data import FEATURE_COLUMNS, read_dataset
from .metrics import classification_metrics
from .plots import plot_evaluation
from .splits import select_partition, verify_manifest


def evaluate_run(data_path: Path, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    output = run_dir/'evaluation'
    if output.exists():
        raise FileExistsError('This run already has an evaluation attempt. Do not reuse the test set for tuning.')
    model,metadata = load_bundle(run_dir)
    frame = read_dataset(data_path,labeled=True)
    manifest = json.loads((run_dir/'split.json').read_text())
    verify_manifest(manifest,data_path,frame,TrainingConfig(**metadata['config']))
    test = select_partition(frame,manifest['ids'],'test')
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:,list(model.classes_).index(1)]
    metrics = {**classification_metrics(test.Churn.to_numpy(),probabilities,metadata['threshold']),
               'run_id':metadata['run_id'], 'data_sha256':metadata['source_sha256'],
               'model_sha256':metadata['model_sha256'], 'evaluated_utc':datetime.now(timezone.utc).isoformat()}
    output.mkdir(exist_ok=False)
    write_json_new(output/'metrics.json',metrics)
    plot_evaluation(test.Churn.to_numpy(),probabilities,metadata['threshold'],output)
    with (output/'COMPLETE').open('x') as stream:
        stream.write('Frozen test evaluation complete.\n')
    logging.info('Test AP %.4f; ROC-AUC %.4f; F1 %.4f on %d rows.',
                 metrics['average_precision'],metrics['roc_auc'],metrics['f1'],len(test))
    return metrics
