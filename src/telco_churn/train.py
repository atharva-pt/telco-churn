"""Select models using training CV; choose decisions using validation only."""
import json
import logging
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from .artifacts import save_bundle
from .config import TrainingConfig
from .data import FEATURE_COLUMNS, read_dataset
from .metrics import choose_threshold, classification_metrics
from .preprocessing import build_pipeline
from .splits import select_partition, verify_manifest


def train_run(data_path: Path, config: TrainingConfig, split_manifest: Path, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f'Run directory already exists: {run_dir}')
    frame = read_dataset(data_path,labeled=True)
    manifest = json.loads(Path(split_manifest).read_text())
    verify_manifest(manifest,data_path,frame,config)
    train = select_partition(frame,manifest['ids'],'train')
    validation = select_partition(frame,manifest['ids'],'validation')
    if train.Churn.value_counts().min() < config.cv_folds:
        raise ValueError('Training class counts are too small for the requested CV folds.')
    dummy = build_pipeline('dummy',config.seed).named_steps['model']
    logistic = build_pipeline('logistic',config.seed).named_steps['model']
    forest = build_pipeline('forest',config.seed).named_steps['model']
    grid = [
        {'model':[dummy]},
        {'model':[logistic], 'model__C':list(config.logistic_c)},
        {'model':[forest], 'model__n_estimators':[config.forest_trees],
         'model__max_depth':list(config.forest_depths), 'model__min_samples_leaf':list(config.forest_min_leaf)},
    ]
    cv = StratifiedKFold(n_splits=config.cv_folds,shuffle=True,random_state=config.seed)
    search = GridSearchCV(build_pipeline('logistic',config.seed), grid, scoring='average_precision',
                          cv=cv, refit=True, n_jobs=1, error_score='raise')
    logging.info('Comparing %d candidates across %d folds on %d training rows.',
                 len(ParameterGrid(grid)),config.cv_folds,len(train))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try:
            search.fit(train[FEATURE_COLUMNS],train.Churn)
        except ConvergenceWarning as exc:
            raise ValueError('Model did not converge. Inspect training scaling/settings before evaluation.') from exc
    model = search.best_estimator_
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:,list(model.classes_).index(1)]
    threshold = choose_threshold(validation.Churn.to_numpy(),probabilities,config.threshold_policy)
    results = []
    for i,params in enumerate(search.cv_results_['params']):
        results.append({'model':type(params['model']).__name__,
                        'parameters':{k:v for k,v in params.items() if k != 'model'},
                        'mean_average_precision':float(search.cv_results_['mean_test_score'][i]),
                        'std_average_precision':float(search.cv_results_['std_test_score'][i]),
                        'rank':int(search.cv_results_['rank_test_score'][i]),
                        'fold_scores':[float(search.cv_results_[f'split{k}_test_score'][i]) for k in range(config.cv_folds)]})
    validation_report = {'policy':config.threshold_policy,
                         'reference':classification_metrics(validation.Churn.to_numpy(),probabilities,0.5),
                         'selected':classification_metrics(validation.Churn.to_numpy(),probabilities,threshold)}
    metadata = {'threshold':threshold, 'config':config.to_dict(),
                'source_sha256':manifest['source_sha256'], 'selected_candidate':results[search.best_index_],
                'training_rows':len(train), 'validation_rows':len(validation)}
    save_bundle(run_dir,model,metadata,{'split.json':manifest,'cv_results.json':results,'validation.json':validation_report})
    logging.info('Saved %s with threshold %.6f to %s.',type(model.named_steps['model']).__name__,threshold,run_dir)
    return json.loads((run_dir/'metadata.json').read_text())
