import json
import tempfile
import unittest
from pathlib import Path

from tests.fixtures import make_frame
from telco_churn.config import TrainingConfig, load_config
from telco_churn.data import read_dataset, validate_frame
from telco_churn.splits import create_manifest, make_split_ids, select_partition, verify_manifest
from telco_churn.plots import explore_training


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_and_exhaustive(self):
        frame = validate_frame(make_frame(), labeled=True)
        ids = make_split_ids(frame, TrainingConfig())
        self.assertEqual(ids, make_split_ids(frame.iloc[::-1], TrainingConfig()))
        self.assertEqual([len(ids[k]) for k in ('train','validation','test')], [60,20,20])
        self.assertEqual(len(set(sum(ids.values(), []))), 100)
        self.assertEqual(set(sum(ids.values(), [])), set(frame.customerID))
        for name in ids:
            self.assertEqual(set(select_partition(frame, ids, name).Churn), {0,1})

    def test_invalid_configs(self):
        for values in [{'seed':True}, {'cv_folds':1}, {'test_fraction':float('nan')},
                       {'test_fraction':0.8,'validation_fraction':0.2},
                       {'threshold_policy':'bad'}, {'logistic_c':[0]}, {'forest_trees':False},
                       {'forest_depths':[]}, {'forest_min_leaf':[1.2]}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig(**values)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'config.json'
            p.write_text('{"unknown": 1}')
            with self.assertRaises(ValueError): load_config(p)

    def test_single_class_or_small_input_fails(self):
        for raw in [make_frame(2), make_frame().assign(Churn='Yes')]:
            with self.assertRaises(ValueError):
                make_split_ids(validate_frame(raw, labeled=True), TrainingConfig())

    def test_manifest_is_bound_to_data_config_and_membership(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input.csv'
            make_frame().to_csv(path, index=False)
            frame = read_dataset(path, labeled=True)
            config = TrainingConfig()
            manifest = create_manifest(path, frame, config)
            verify_manifest(manifest, path, frame, config)
            for mode in ('overlap','missing','extra','config','source'):
                bad = json.loads(json.dumps(manifest))
                if mode == 'overlap': bad['ids']['test'][0] = bad['ids']['train'][0]
                elif mode == 'missing': bad['ids']['train'].pop()
                elif mode == 'extra': bad['ids']['train'].append('unknown')
                elif mode == 'config': bad['config']['seed'] = 123
                else: bad['source_sha256'] = '0'*64
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    verify_manifest(bad, path, frame, config)

    def test_exploration_uses_only_given_partition(self):
        frame = validate_frame(make_frame(), labeled=True)
        train = select_partition(frame, make_split_ids(frame, TrainingConfig()), 'train')
        with tempfile.TemporaryDirectory() as d:
            explore_training(train, Path(d))
            summary = json.loads((Path(d)/'summary.json').read_text())
            self.assertEqual(summary['rows'], 60)
            self.assertEqual(sum(summary['class_counts'].values()), 60)
            self.assertEqual(len(list(Path(d).glob('*.png'))), 3)
