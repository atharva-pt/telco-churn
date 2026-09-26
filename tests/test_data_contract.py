import tempfile
import unittest
from pathlib import Path

import pandas as pd
from tests.fixtures import make_frame
from telco_churn.data import FEATURE_COLUMNS, SchemaError, normalize_features, read_dataset, validate_frame


class DataContractTests(unittest.TestCase):
    def test_blank_charge_and_target_mapping(self):
        raw = make_frame()
        raw.loc[0, 'TotalCharges'] = ' '
        clean = validate_frame(raw, labeled=True)
        self.assertTrue(pd.isna(clean.loc[0, 'TotalCharges']))
        self.assertEqual(clean.Churn.tolist()[:2], [0, 1])
        self.assertEqual(raw.loc[0, 'TotalCharges'], ' ')

    def test_reordered_columns_are_canonical(self):
        frame = make_frame()
        pd.testing.assert_frame_equal(validate_frame(frame, labeled=True),
                                      validate_frame(frame.iloc[:, ::-1], labeled=True))

    def test_invalid_values_fail_with_field_name(self):
        for column, value in [('MonthlyCharges', 'oops'), ('MonthlyCharges', '-1'),
                              ('TotalCharges', 'NaN'), ('TotalCharges', 'inf'),
                              ('tenure', '1.5'), ('gender', 'unknown'), ('SeniorCitizen', '2'),
                              ('Churn', 'maybe'), ('Churn', ''), ('Contract', ''), ('customerID', '')]:
            with self.subTest(column=column, value=value):
                raw = make_frame()
                raw.loc[0, column] = value
                with self.assertRaisesRegex(SchemaError, column):
                    validate_frame(raw, labeled=True)

    def test_invalid_shapes_and_ids_fail(self):
        frame = make_frame()
        bad_frames = [frame.iloc[:0], frame.drop(columns='tenure'), frame.assign(extra=1),
                      pd.concat([frame, frame.iloc[:1]]), frame.rename(columns={'tenure': 'gender'})]
        for bad in bad_frames:
            with self.subTest(shape=bad.shape), self.assertRaises(SchemaError):
                validate_frame(bad, labeled=True)
        with self.assertRaises(SchemaError):
            validate_frame(frame, labeled=False)

    def test_csv_structure_and_leading_zero_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'input.csv'
            make_frame().to_csv(path, index=False, encoding='utf-8-sig')
            self.assertEqual(read_dataset(path, labeled=True).customerID.iloc[0], '00000')
            for text in ['', 'a,a\n1,2\n', 'a,b\n1,2,3\n', ','.join(make_frame().columns) + '\n']:
                path.write_text(text)
                with self.subTest(text=text[:20]), self.assertRaises(SchemaError):
                    read_dataset(path, labeled=True)

    def test_whitespace_normalization_is_idempotent(self):
        frame = make_frame()[FEATURE_COLUMNS]
        frame.loc[0, 'Contract'] = ' One year '
        frame.loc[0, 'TotalCharges'] = ''
        clean = normalize_features(frame)
        pd.testing.assert_frame_equal(clean, normalize_features(clean))


if __name__ == '__main__':
    unittest.main()
