"""Tests for metrics, model accounting and the reported results."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuropulse import metrics as M  # noqa: E402
from neuropulse import models  # noqa: E402
from neuropulse import reported_results as R  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_confusion_and_scores(self):
        y = [0, 0, 1, 1, 2, 2]
        p = [0, 1, 1, 1, 2, 0]
        cm = M.confusion_matrix(y, p, 3)
        np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        s = M.summary_scores(cm)
        self.assertAlmostEqual(s["accuracy"], 4 / 6)
        self.assertAlmostEqual(s["balanced_accuracy"], np.mean([0.5, 1.0, 0.5]))

    def test_confusion_rejects_bad_labels(self):
        with self.assertRaises(ValueError):
            M.confusion_matrix([0, 3], [0, 1], 3)
        with self.assertRaises(ValueError):
            M.confusion_matrix([0, 1], [0], 3)

    def test_undefined_precision_is_zero_not_nan(self):
        cm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
        p, r, f = M.per_class_prf(cm)
        self.assertTrue(np.isfinite(p).all() and np.isfinite(r).all() and np.isfinite(f).all())
        self.assertEqual(f[2], 0.0)

    def test_clopper_pearson_known_values(self):
        lo, hi = M.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3084, places=3)
        lo, hi = M.clopper_pearson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.6916, places=3)
        with self.assertRaises(ValueError):
            M.clopper_pearson(11, 10)

    def test_intervals_contain_point_estimate(self):
        for k in (300, 361, 366, 367, 375):
            for lo, hi in (M.clopper_pearson(k, 375), M.wilson(k, 375)):
                self.assertLessEqual(lo, k / 375 + 1e-12)
                self.assertGreaterEqual(hi, k / 375 - 1e-12)
                self.assertTrue(0.0 <= lo <= hi <= 1.0 + 1e-12)

    def test_bootstrap_interval_brackets_accuracy(self):
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(5), 75)
        pred = y.copy()
        wrong = rng.choice(375, 9, replace=False)
        pred[wrong] = (pred[wrong] + 1) % 5
        lo, hi = M.stratified_bootstrap_accuracy(y, pred, replicates=500, seed=1)
        self.assertLess(lo, 366 / 375)
        self.assertGreater(hi, 366 / 375)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)

    def test_ece_perfect_and_overconfident(self):
        y = np.array([0, 1, 2, 3])
        perfect = np.eye(4)
        self.assertAlmostEqual(M.expected_calibration_error(perfect, y), 0.0)
        wrong = np.eye(4)[[1, 2, 3, 0]]
        self.assertAlmostEqual(M.expected_calibration_error(wrong, y), 1.0)

    def test_brier(self):
        y = np.array([0, 1])
        self.assertAlmostEqual(M.multiclass_brier(np.eye(2), y), 0.0)
        self.assertAlmostEqual(M.multiclass_brier(np.eye(2)[[1, 0]], y), 2.0)
        self.assertAlmostEqual(M.multiclass_brier(np.full((2, 2), 0.5), y), 0.5)

    def test_mcnemar(self):
        y = np.zeros(20, dtype=int)
        a = y.copy()
        b = y.copy()
        b[:6] = 1  # model B is wrong on 6 examples where A is right
        n_a_only, n_b_only, p = M.mcnemar_exact(y, a, b)
        self.assertEqual((n_a_only, n_b_only), (6, 0))
        self.assertAlmostEqual(p, 2 * 0.5 ** 6)
        self.assertEqual(M.mcnemar_exact(y, a, a)[2], 1.0)

    def test_confidence_floor_matches_manuscript_arithmetic(self):
        floor = M.confidence_floor_for_errors(R.CALIBRATION["mean_confidence"], R.N_TEST, 9)
        self.assertAlmostEqual(floor, 0.829, places=3)


class ModelAccountingTests(unittest.TestCase):
    def test_cnn_attention_parameter_total(self):
        self.assertEqual(models.cnn_attention_parameter_count(), R.CNN_ATTENTION_PARAMETERS)
        self.assertEqual(models.cnn_attention_parameter_count(), 42821)

    def test_layer_breakdown(self):
        counts = {name: p for name, _, p in models.cnn_attention_layers()}
        self.assertEqual(counts["Conv1D, 64 filters, kernel 2, same padding, ReLU"], 960)
        self.assertEqual(counts["Conv1D, 128 filters, kernel 2, same padding, ReLU"], 16512)
        self.assertEqual(counts["Layer normalization"], 256)
        self.assertEqual(counts["Dense 128, ReLU"], 16512)
        self.assertEqual(counts["Dense 64, ReLU"], 8256)
        self.assertEqual(counts["Dense 5, softmax"], 325)
        self.assertEqual(counts["Self-attention, Q = K = V (no projections)"], 0)

    def test_mlp_count_and_attention_shape(self):
        self.assertEqual(models.mlp_parameter_count(), 12293)
        self.assertEqual(models.attention_matrix_shape(), (4, 4))

    def test_svm_pipeline_fits(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (40, 28)), rng.normal(3, 1, (40, 28))])
        y = np.repeat([0, 1], 40)
        clf = models.build_rbf_svm().fit(X, y)
        self.assertGreater(clf.score(X, y), 0.95)
        self.assertEqual(clf.named_steps["svc"].kernel, "rbf")
        self.assertEqual(clf.named_steps["svc"].C, 1.0)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_keras_model_matches_analytic_count(self):
        model = models.build_cnn_attention()
        self.assertEqual(model.count_params(), R.CNN_ATTENTION_PARAMETERS)
        self.assertEqual(model.output_shape, (None, 5))


class ReportedResultsTests(unittest.TestCase):
    def test_verification_script_passes(self):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_reported_results.py")],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("All consistency checks passed.", proc.stdout)

    def test_every_model_has_only_angry_fear_errors_within_support(self):
        for model, (a2f, f2a) in R.ERRORS.items():
            self.assertLessEqual(a2f, R.N_PER_CLASS_TEST, model)
            self.assertLessEqual(f2a, R.N_PER_CLASS_TEST, model)


if __name__ == "__main__":
    unittest.main()
