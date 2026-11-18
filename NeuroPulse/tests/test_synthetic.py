"""Tests for the reference generator used in the supplementary experiments."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neuropulse import synthetic as S  # noqa: E402
from neuropulse.contract import DEFAULT_SCHEMA  # noqa: E402


class GeneratorTests(unittest.TestCase):
    def test_shape_columns_and_balance(self):
        df = S.make_dataset(n_per_class=40, seed=3)
        self.assertEqual(df.shape, (200, 29))
        self.assertEqual(list(df.columns[:28]), list(DEFAULT_SCHEMA.columns))
        self.assertEqual(df["label"].value_counts().to_dict(), {c: 40 for c in S.CLASSES})
        self.assertFalse(df.isna().any().any())

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        a, b, c = S.make_dataset(30, seed=5), S.make_dataset(30, seed=5), S.make_dataset(30, seed=6)
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(c))

    def test_band_power_floor(self):
        df = S.make_dataset(200, separation=0.5, noise_sd=3.0, seed=1)
        self.assertGreaterEqual(df[list(S.band_power_columns())].to_numpy().min(), S.BAND_FLOOR)

    def test_std_descriptors_are_positive(self):
        df = S.make_dataset(100, seed=2)
        std_cols = [c for c in df.columns if c.endswith("_std")]
        self.assertEqual(len(std_cols), 4)
        self.assertTrue((df[std_cols] > 0).all().all())

    def test_noise_features_do_not_depend_on_class(self):
        """Class means of the eight label-independent columns agree within sampling error."""
        df = S.make_dataset(2000, seed=4)
        for col in S.noise_columns():
            means = df.groupby("label")[col].mean()
            self.assertLess(means.max() - means.min(), 0.12, col)

    def test_band_features_do_depend_on_class(self):
        df = S.make_dataset(2000, seed=4)
        spread = df.groupby("label")["AF8_gamma"].mean()
        self.assertGreater(spread.max() - spread.min(), 1.0)

    def test_separation_scales_class_distance(self):
        d = lambda s: np.abs(S.class_means("HAPPY", s) - S.class_means("SAD", s)).sum()
        self.assertAlmostEqual(d(2.0), 2 * d(1.0), places=9)

    def test_angry_and_fear_are_the_closest_pair_of_classes(self):
        dist = {(a, b): np.abs(S.class_means(a) - S.class_means(b)).sum()
                for i, a in enumerate(S.CLASSES) for b in S.CLASSES[i + 1:]}
        self.assertEqual(min(dist, key=dist.get), ("ANGRY", "FEAR"))

    def test_counts_of_feature_groups(self):
        self.assertEqual(len(S.band_power_columns()), 20)
        self.assertEqual(len(S.noise_columns()), 8)


class OrderUtilityTests(unittest.TestCase):
    def test_column_index_is_a_permutation_and_identity_by_default(self):
        import run_supplementary_experiments as R

        ident = R.column_index()
        np.testing.assert_array_equal(ident, np.arange(28))
        for perm in R.DESCRIPTOR_PERMS.values():
            idx = R.column_index(desc_perm=perm)
            self.assertEqual(sorted(idx.tolist()), list(range(28)))
        for perm in R.CHANNEL_PERMS.values():
            idx = R.column_index(ch_perm=perm)
            self.assertEqual(sorted(idx.tolist()), list(range(28)))

    def test_descriptor_permutation_acts_within_each_channel_group(self):
        import run_supplementary_experiments as R

        idx = R.column_index(desc_perm=[6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(idx[:7].tolist(), [6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(idx[7:14].tolist(), [13, 12, 11, 10, 9, 8, 7])

    def test_channel_permutation_moves_whole_groups(self):
        import run_supplementary_experiments as R

        idx = R.column_index(ch_perm=[1, 0, 2, 3])
        self.assertEqual(idx[:7].tolist(), list(range(7, 14)))
        self.assertEqual(idx[7:14].tolist(), list(range(0, 7)))


if __name__ == "__main__":
    unittest.main()
