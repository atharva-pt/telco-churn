"""Explicit, validated experiment settings; no hidden global training state."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    cv_folds: int = 5
    threshold_policy: str = 'max_f1'
    logistic_c: tuple[float, ...] = (0.1, 1.0, 10.0)
    forest_trees: int = 200
    forest_depths: tuple[int | None, ...] = (6, None)
    forest_min_leaf: tuple[int, ...] = (1, 5)

    def __post_init__(self):
        for name in ('seed', 'cv_folds', 'forest_trees'):
            value = getattr(self, name)
            minimum = 0 if name == 'seed' else (2 if name == 'cv_folds' else 1)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name}: expected integer >= {minimum}.')
        if self.seed > 2**32 - 1:
            raise ValueError('seed: must be at most 2**32 - 1.')
        for name in ('test_fraction', 'validation_fraction'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name}: expected a finite fraction between 0 and 1.')
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError('Split fractions must leave a nonempty training partition.')
        if self.threshold_policy not in ('max_f1', 'fixed_0_5'):
            raise ValueError('threshold_policy: use max_f1 or fixed_0_5.')
        for name in ('logistic_c', 'forest_depths', 'forest_min_leaf'):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f'{name}: expected a nonempty list.')
            for value in values:
                if name == 'forest_depths' and value is None:
                    continue
                allowed_types = (int, float) if name == 'logistic_c' else (int,)
                if type(value) not in allowed_types or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name}: invalid search value.')
            object.__setattr__(self, name, tuple(values))

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def load_config(path: Path) -> TrainingConfig:
    try:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError('Configuration must be a JSON object.')
        return TrainingConfig(**payload)
    except TypeError as exc:
        raise ValueError('Configuration contains unknown keys or invalid value types.') from exc
