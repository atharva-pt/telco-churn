"""Run files are explicit, inspectable, and never silently overwritten."""
import hashlib
import json
import importlib.metadata
import math
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path

from sklearn.pipeline import Pipeline

from .data import CATEGORIES, FEATURE_COLUMNS, NUMERIC_COLUMNS

REPORT_NAMES = {'split.json', 'cv_results.json', 'validation.json'}


class ArtifactError(ValueError):
    """A run is incomplete, incompatible, or has been modified."""


def file_sha256(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def environment_versions() -> dict[str, str]:
    versions = {'python': platform.python_version()}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata['Name'].lower().replace('_', '-')
        if name != 'telco-churn':
            versions[name] = distribution.version
    return dict(sorted(versions.items()))


def source_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(source_dir).rglob('*.py')):
        digest.update(path.relative_to(source_dir).as_posix().encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def _schema() -> dict:
    return {'features': FEATURE_COLUMNS, 'numeric': NUMERIC_COLUMNS, 'categories': CATEGORIES}


def save_bundle(run_dir: Path, pipeline: Pipeline, metadata: dict, reports: dict[str, object]) -> None:
    """Write a trusted local model; COMPLETE is committed only after all files."""
    if set(reports) != REPORT_NAMES:
        raise ValueError('A bundle requires exactly split.json, cv_results.json, validation.json.')
    threshold = metadata.get('threshold')
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid decision threshold.')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir/'pipeline.pkl').open('xb') as stream:
        pickle.dump(pipeline, stream, protocol=5)
    for name, value in reports.items():
        write_json_new(run_dir/name, value)
    saved = {**metadata, 'format_version': 1, 'run_id':run_dir.name,
             'created_utc':datetime.now(timezone.utc).isoformat(), 'schema':_schema(),
             'environment':environment_versions(),
             'code_sha256':source_fingerprint(Path(__file__).parent),
             'model_sha256':file_sha256(run_dir/'pipeline.pkl'),
             'report_sha256':{name:file_sha256(run_dir/name) for name in sorted(REPORT_NAMES)}}
    write_json_new(run_dir/'metadata.json',saved)
    with (run_dir/'COMPLETE').open('x') as stream:
        stream.write('Model bundle complete.\n')


def load_bundle(run_dir: Path) -> tuple[Pipeline, dict]:
    """Only load artifacts you trust. Hashes do not make pickle safe."""
    run_dir = Path(run_dir)
    try:
        if not (run_dir/'COMPLETE').is_file():
            raise ArtifactError('Model bundle is incomplete (missing COMPLETE).')
        metadata = json.loads((run_dir/'metadata.json').read_text())
        if metadata['format_version'] != 1 or metadata['schema'] != _schema():
            raise ArtifactError('Model schema/format is incompatible.')
        current = environment_versions()
        recorded = dict(metadata['environment'])
        # Patch-level Python differences are allowed; libraries must match exactly.
        current['python'] = '.'.join(current['python'].split('.')[:2])
        recorded['python'] = '.'.join(recorded['python'].split('.')[:2])
        if current != recorded:
            raise ArtifactError('Environment mismatch. Recreate the environment from requirements.lock.txt.')
        if metadata['code_sha256'] != source_fingerprint(Path(__file__).parent):
            raise ArtifactError('Package source differs from the version that saved this model.')
        threshold = metadata['threshold']
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ArtifactError('Invalid saved threshold.')
        if file_sha256(run_dir/'pipeline.pkl') != metadata['model_sha256']:
            raise ArtifactError('Model checksum mismatch.')
        if set(metadata['report_sha256']) != REPORT_NAMES:
            raise ArtifactError('Missing bundle report checksums.')
        for name in REPORT_NAMES:
            if file_sha256(run_dir/name) != metadata['report_sha256'][name]:
                raise ArtifactError(f'Bundle report checksum mismatch: {name}.')
        with (run_dir/'pipeline.pkl').open('rb') as stream:
            model = pickle.load(stream)
        if not isinstance(model, Pipeline) or set(model.classes_) != {0,1}:
            raise ArtifactError('Saved object is not a fitted binary-classification pipeline.')
        return model, metadata
    except ArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, pickle.UnpicklingError) as exc:
        raise ArtifactError('Unable to read a valid, complete model bundle.') from exc
