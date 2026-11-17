#!/usr/bin/env python
"""Check that every number printed in the manuscript is consistent with the
reported counts, and print the derived quantities used in the paper.

Run from the repository root:  python scripts/verify_reported_results.py
Exit status is non-zero when any check fails.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuropulse import metrics as M  # noqa: E402
from neuropulse import models  # noqa: E402
from neuropulse import reported_results as R  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def confusion_for(model):
    a2f, f2a = R.ERRORS[model]
    cm = np.diag([R.N_PER_CLASS_TEST] * 5).astype(int)
    cm[0, 0] -= a2f
    cm[0, 2] += a2f
    cm[2, 2] -= f2a
    cm[2, 0] += f2a
    return cm


print("== Split sizes ==")
check("train + validation + test = total", R.N_TRAIN + R.N_VALIDATION + R.N_TEST == R.N_TOTAL)
check("test size = 5 x per-class support", R.N_TEST == 5 * R.N_PER_CLASS_TEST)

print("\n== Single-run accuracy from error counts ==")
for m in R.MODELS:
    cm = confusion_for(m)
    n_ok = int(np.trace(cm))
    acc = 100 * n_ok / R.N_TEST
    check(f"{m}: accuracy {acc:.3f} matches printed {R.SINGLE_RUN_ACCURACY[m]:.3f}",
          abs(acc - R.SINGLE_RUN_ACCURACY[m]) < 5e-4, f"({n_ok}/{R.N_TEST})")
    check(f"{m}: rows sum to class support", bool((cm.sum(axis=1) == R.N_PER_CLASS_TEST).all()))

print("\n== CNN-attention per-class metrics (Table 5) ==")
cm = confusion_for("CNN-attention")
P, Rc, F = M.per_class_prf(cm)
for k, name in enumerate(R.CLASSES):
    ref = R.REPORTED_CNN_ATT_PER_CLASS[name]
    ok = all(abs(round(v, 4) - r) < 1e-4 for v, r in zip((P[k], Rc[k], F[k]), ref))
    check(f"{name}: P/R/F1 = {P[k]:.4f}/{Rc[k]:.4f}/{F[k]:.4f}", ok)
s = M.summary_scores(cm)
for key, rep in (("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1"),
                 ("balanced_accuracy", "balanced_accuracy")):
    check(f"macro {rep} = {s[key]:.4f}", abs(round(s[key], 4) - R.REPORTED_MACRO[rep]) < 1e-4)

print("\n== Bootstrap interval endpoints are attainable accuracies (multiples of 1/375) ==")
for v in R.BOOTSTRAP_ACCURACY_CI:
    check(f"{v:.2f}% = {round(v * R.N_TEST / 100)}/{R.N_TEST}", abs(v - 100 * round(v * R.N_TEST / 100) / R.N_TEST) < 0.006)

print("\n== Perturbation and ablation accuracies correspond to whole records ==")
for table in (R.ORDER_SENSITIVITY, R.CHANNEL_ABLATION):
    for name, v in table.items():
        n_ok = round(v * R.N_TEST / 100)
        check(f"{name}: {v}% = {n_ok}/{R.N_TEST}", abs(100 * n_ok / R.N_TEST - v) < 0.006)
check("every single-row-removal accuracy is below the 97.60% reference",
      all(v < R.ORDER_SENSITIVITY["Unpermuted"] for v in R.CHANNEL_ABLATION.values()))

print("\n== Calibration arithmetic ==")
cal = R.CALIBRATION
gap = cal["mean_confidence"] - R.SINGLE_RUN_ACCURACY["CNN-attention"] / 100
check(f"ECE {cal['ece_10bin']} >= |mean confidence - accuracy| = {gap:.4f}", cal["ece_10bin"] >= gap - 1e-9)
floor = M.confidence_floor_for_errors(cal["mean_confidence"], R.N_TEST, sum(R.ERRORS["CNN-attention"]))
print(f"       derived: mean top score of the 9 wrong predictions is at least {floor:.3f}")
check("lower bound on error confidence is below 1", floor < 1.0)

print("\n== Seed summary ==")
mean_gap = R.THREE_SEED["RBF-SVM"]["acc_mean"] - R.THREE_SEED["CNN-attention"]["acc_mean"]
print(f"       SVM minus CNN-attention mean accuracy = {mean_gap:.3f} pp = {mean_gap * R.N_TEST / 100:.2f} test records")
check("SVM seed mean equals its single-run accuracy", R.THREE_SEED["RBF-SVM"]["acc_mean"] == R.SINGLE_RUN_ACCURACY["RBF-SVM"])
print(f"       one test record = {100 / R.N_TEST:.4f} pp (CNN-attention SD = {R.THREE_SEED['CNN-attention']['acc_sd']} pp)")

print("\n== Derived per-class metrics for all models (cf. Table 7) ==")
print("model, class, precision, recall, F1")
for m in R.MODELS:
    P, Rc, F = M.per_class_prf(confusion_for(m))
    for k in (0, 2):
        print(f"{m}, {R.CLASSES[k]}, {P[k]:.4f}, {Rc[k]:.4f}, {F[k]:.4f}")
    sc = M.summary_scores(confusion_for(m))
    print(f"{m}, macro, P={sc['macro_precision']:.4f}, R={sc['macro_recall']:.4f}, F1={sc['macro_f1']:.4f}")

print("\n== Exact (Clopper-Pearson) and Wilson 95% intervals from test counts (Table 7) ==")
for m in R.MODELS:
    n_ok = int(np.trace(confusion_for(m)))
    cp = M.clopper_pearson(n_ok, R.N_TEST)
    wl = M.wilson(n_ok, R.N_TEST)
    print(f"{m}: {n_ok}/{R.N_TEST}  CP [{100 * cp[0]:.2f}, {100 * cp[1]:.2f}]  Wilson [{100 * wl[0]:.2f}, {100 * wl[1]:.2f}]")

print("\n== Parameter accounting ==")
total = models.cnn_attention_parameter_count()
check(f"CNN-attention total parameters = {total:,}", total == R.CNN_ATTENTION_PARAMETERS)
print(f"       MLP (28-128-64-5) parameters = {models.mlp_parameter_count():,}")
extra = 3 * (128 * 128 + 128)
print(f"       learned Q/K/V projections of width 128 would add {extra:,} parameters, "
      f"giving {total + extra:,}")
check("reported total excludes learned Q/K/V projections", total + extra != R.CNN_ATTENTION_PARAMETERS)

print("\n== Below-chance accuracy under the mixed descriptor order ==")
from scipy import stats  # noqa: E402

n_ok = round(R.ORDER_SENSITIVITY["Descriptors mixed"] * R.N_TEST / 100)
p = stats.binomtest(n_ok, R.N_TEST, 0.2, alternative="less").pvalue
print(f"       {n_ok}/{R.N_TEST} correct against a chance rate of 0.20: one-sided exact p = {p:.2e}")

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("All consistency checks passed.")
