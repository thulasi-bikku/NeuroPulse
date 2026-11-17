"""Tests of the dashboard input contract (manuscript Section 7.3, Table 9)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuropulse import contract as C  # noqa: E402

SCHEMA = C.DEFAULT_SCHEMA
CLASSES = C.DEFAULT_CLASSES


def make_frame(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(1.0, 0.5, size=(n, SCHEMA.n_features)), columns=SCHEMA.columns)


def fitted_scaler(frame):
    return StandardScaler().fit(frame[list(SCHEMA.columns)])


def fixed_model(seed=7):
    """Deterministic stand-in for a trained network: softmax of a fixed linear map."""
    w = np.random.default_rng(seed).normal(size=(SCHEMA.n_features, len(CLASSES)))

    def predict_proba(tensor):
        assert tensor.shape[1:] == SCHEMA.shape
        return C.softmax(tensor.reshape(tensor.shape[0], -1) @ w)

    return predict_proba


class SchemaTests(unittest.TestCase):
    def test_canonical_layout(self):
        self.assertEqual(SCHEMA.n_features, 28)
        self.assertEqual(SCHEMA.shape, (4, 7))
        self.assertEqual(SCHEMA.columns[0], "TP9_delta")
        self.assertEqual(SCHEMA.columns[6], "TP9_std")
        self.assertEqual(SCHEMA.columns[7], "AF7_delta")
        self.assertEqual(SCHEMA.columns[-1], "TP10_std")

    def test_reshape_places_channel_groups_on_first_axis(self):
        frame = make_frame(3)
        scaler = fitted_scaler(frame)
        values, valid = C.to_numeric_matrix(frame, SCHEMA)
        self.assertTrue(valid.all())
        tensor = C.scale_and_reshape(values, scaler, SCHEMA)
        flat = scaler.transform(pd.DataFrame(values, columns=SCHEMA.columns))
        self.assertEqual(tensor.shape, (3, 4, 7))
        np.testing.assert_allclose(tensor[:, 1, 0], flat[:, 7])   # AF7_delta
        np.testing.assert_allclose(tensor[:, 3, 6], flat[:, 27])  # TP10_std


class ParsingTests(unittest.TestCase):
    def test_valid_csv_round_trip(self):
        frame = make_frame(4)
        parsed = C.read_table_from_text(frame.to_csv(index=False))
        self.assertEqual(list(parsed.columns), list(SCHEMA.columns))
        self.assertEqual(len(parsed), 4)

    def test_empty_input(self):
        for text in ("", "   \n  "):
            with self.assertRaises(C.EmptyInputError):
                C.read_table_from_text(text)

    def test_header_only_table_is_empty_for_prediction(self):
        parsed = C.read_table_from_text(",".join(SCHEMA.columns) + "\n")
        with self.assertRaises(C.EmptyInputError):
            C.predict_batch(parsed, fitted_scaler(make_frame()), fixed_model())

    def test_duplicate_header_is_rejected(self):
        header = list(SCHEMA.columns)
        header[5] = header[4]
        text = ",".join(header) + "\n" + ",".join(["1"] * 28) + "\n"
        with self.assertRaises(C.SchemaError):
            C.read_table_from_text(text)

    def test_malformed_csv_gives_controlled_error(self):
        text = ",".join(SCHEMA.columns) + "\n" + ",".join(["1"] * 28) + "\n" + ",".join(["1"] * 40) + "\n"
        with self.assertRaises(C.ParseError):
            C.read_table_from_text(text)

    def test_undecodable_file_gives_controlled_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_bytes(b"\xff\xfe\x00\x80\x81")
            with self.assertRaises(C.ParseError):
                C.read_table(path)

    def test_missing_file_gives_controlled_error(self):
        with self.assertRaises(C.ParseError):
            C.read_table("/nonexistent/path/table.csv")


class ValidationTests(unittest.TestCase):
    def test_missing_required_column(self):
        frame = make_frame().drop(columns=["AF8_beta"])
        with self.assertRaises(C.SchemaError) as ctx:
            C.to_numeric_matrix(frame, SCHEMA)
        self.assertIn("AF8_beta", str(ctx.exception))

    def test_extra_columns_are_ignored(self):
        frame = make_frame()
        frame["label"] = "CALM"
        values, valid = C.to_numeric_matrix(frame, SCHEMA)
        self.assertEqual(values.shape, (len(frame), 28))
        self.assertTrue(valid.all())

    def test_column_order_is_irrelevant(self):
        frame = make_frame(8)
        scaler, model = fitted_scaler(frame), fixed_model()
        shuffled = frame[list(np.random.default_rng(3).permutation(frame.columns))]
        a = C.predict_batch(frame, scaler, model)
        b = C.predict_batch(shuffled, scaler, model)
        pd.testing.assert_frame_equal(a.predictions, b.predictions)

    def test_non_finite_and_non_numeric_rows_are_masked(self):
        frame = make_frame(6).astype(object)
        frame.loc[1, "TP9_delta"] = np.nan
        frame.loc[2, "AF7_theta"] = "abc"
        frame.loc[3, "AF8_gamma"] = np.inf
        frame.loc[4, "TP10_std"] = "-inf"
        values, valid = C.to_numeric_matrix(frame, SCHEMA)
        self.assertEqual(valid.tolist(), [True, False, False, False, False, True])
        self.assertTrue(np.isfinite(values[valid]).all())

    def test_row_index_validation(self):
        for bad in (-1, 12, 99, 1.5, "3", None, True):
            with self.assertRaises(C.RowIndexError):
                C.validate_row_index(bad, 12)
        self.assertEqual(C.validate_row_index(0, 12), 0)
        self.assertEqual(C.validate_row_index(np.int64(11), 12), 11)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame()
        self.scaler = fitted_scaler(self.frame)

    def test_compatible_artifacts_pass(self):
        C.check_artifacts(self.scaler, CLASSES, SCHEMA)

    def test_missing_artifacts(self):
        with self.assertRaises(C.ArtifactError):
            C.check_artifacts(None, CLASSES, SCHEMA)
        with self.assertRaises(C.ArtifactError):
            C.check_artifacts(self.scaler, None, SCHEMA)

    def test_wrong_feature_count(self):
        narrow = StandardScaler().fit(self.frame.iloc[:, :20])
        with self.assertRaises(C.ArtifactError):
            C.check_artifacts(narrow, CLASSES, SCHEMA)

    def test_scaler_fitted_with_different_feature_order(self):
        reordered = self.frame[list(reversed(SCHEMA.columns))]
        with self.assertRaises(C.ArtifactError):
            C.check_artifacts(StandardScaler().fit(reordered), CLASSES, SCHEMA)

    def test_different_class_order(self):
        with self.assertRaises(C.ArtifactError):
            C.check_artifacts(self.scaler, ("CALM", "ANGRY", "FEAR", "HAPPY", "SAD"), SCHEMA)

    def test_model_output_with_wrong_shape_is_rejected(self):
        with self.assertRaises(C.ArtifactError):
            C.predict_batch(self.frame, self.scaler, lambda t: np.ones((len(t), 3)))

    def test_model_output_with_nan_is_rejected(self):
        with self.assertRaises(C.ArtifactError):
            C.predict_batch(self.frame, self.scaler, lambda t: np.full((len(t), 5), np.nan))


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame(10)
        self.scaler = fitted_scaler(self.frame)
        self.model = fixed_model()

    def test_scaler_is_reused_not_refitted(self):
        """Uploaded data must be transformed with the training statistics."""
        shifted = self.frame + 50.0
        values, _ = C.to_numeric_matrix(shifted, SCHEMA)
        tensor = C.scale_and_reshape(values, self.scaler, SCHEMA)
        self.assertGreater(tensor.mean(), 10.0)  # a refitted scaler would centre this near zero

    def test_valid_single_row(self):
        row = C.predict_single(self.frame, 4, self.scaler, self.model)
        self.assertIn(row["predicted"], CLASSES)
        self.assertTrue(0 < row["confidence_percent"] <= 100)
        self.assertEqual(int(row["row"]), 4)
        self.assertAlmostEqual(sum(row[f"score_{k}"] for k in CLASSES), 1.0, places=9)

    def test_invalid_single_row_is_reported(self):
        frame = self.frame.copy()
        frame.loc[2, "AF8_alpha"] = np.nan
        with self.assertRaises(C.EmptyInputError):
            C.predict_single(frame, 2, self.scaler, self.model)

    def test_empty_batch(self):
        with self.assertRaises(C.EmptyInputError):
            C.predict_batch(self.frame.iloc[0:0], self.scaler, self.model)
        with self.assertRaises(C.EmptyInputError):
            C.predict_batch(None, self.scaler, self.model)

    def test_all_invalid_batch(self):
        frame = self.frame.copy()
        frame["TP9_delta"] = np.nan
        with self.assertRaises(C.EmptyInputError):
            C.predict_batch(frame, self.scaler, self.model)

    def test_mixed_validity_batch_reports_exclusions(self):
        frame = self.frame.copy()
        frame.loc[[1, 7], "AF7_beta"] = np.nan
        result = C.predict_batch(frame, self.scaler, self.model)
        self.assertEqual((result.n_total, result.n_valid, result.n_invalid), (10, 8, 2))
        self.assertEqual(result.invalid_rows, [1, 7])
        self.assertNotIn(1, result.predictions["row"].tolist())
        self.assertEqual(len(result.predictions), 8)

    def test_softmax_properties(self):
        z = np.array([[1000.0, 1001.0, 999.0, 0.0, -5.0]])
        p = C.softmax(z)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        self.assertEqual(int(p.argmax()), 1)

    def test_batch_summary(self):
        preds = pd.DataFrame({"predicted": ["CALM", "CALM", "FEAR", "SAD"],
                              "confidence_percent": [90.0, 80.0, 70.0, 60.0]})
        s = C.summarise(preds)
        self.assertEqual(s["denominator"], 4)
        self.assertEqual(s["counts"]["CALM"], 2)
        self.assertEqual(s["top_class"], "CALM")
        self.assertAlmostEqual(s["shares_percent"]["CALM"], 50.0)
        self.assertAlmostEqual(sum(s["shares_percent"].values()), 100.0)
        self.assertAlmostEqual(s["mean_confidence_percent"], 75.0)

    def test_summary_tie_breaks_by_class_order(self):
        preds = pd.DataFrame({"predicted": ["SAD", "FEAR"], "confidence_percent": [50.0, 50.0]})
        self.assertEqual(C.summarise(preds)["top_class"], "FEAR")

    def test_summary_of_nothing_is_an_error(self):
        with self.assertRaises(C.EmptyInputError):
            C.summarise(pd.DataFrame({"predicted": [], "confidence_percent": []}))


class HistoryTests(unittest.TestCase):
    def test_round_trip_and_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.db"
            first = C.HistoryStore(path)
            first.add(3, "CALM", 91.5)
            first.add(4, "FEAR", 77.0)
            second = C.HistoryStore(path)  # a new object on the same file, as after an application restart
            rows = second.all()
            self.assertEqual([r[1:] for r in rows], [(3, "CALM", 91.5), (4, "FEAR", 77.0)])
            self.assertTrue(all(r[0].endswith("+00:00") for r in rows))

    def test_values_are_parameterised(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = C.HistoryStore(Path(tmp) / "h.db")
            store.add(0, "CALM'); DROP TABLE predictions;--", 50.0)
            self.assertEqual(len(store.all()), 1)


if __name__ == "__main__":
    unittest.main()
