#!/usr/bin/env python
"""Supplementary controlled experiments on the reference generator.

Six experiments, each over independent replicates (a replicate = a newly
generated 2,500-record dataset and a new stratified 70:15:15 split):

  E1  baseline comparison at the default separation
  E2  difficulty sweep over the class-separation parameter
  E3  negative controls (shuffled labels; label-independent features only)
  E4  feature-order perturbation: evaluate-only versus retrained
  E5  learning curves over the training-set size
  E6  neutralising one channel group at test time

Hyperparameters are fixed in advance and not tuned; the validation partition is
therefore left unused (the MLP uses its own internal early-stopping split). The
CNN-attention model of the main experiment is not part of this study, because
TensorFlow is not required by this package.

Usage:  python scripts/run_supplementary_experiments.py [--replicates 10] [--quick]
Output: docs/supplementary_results.json
"""
import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuropulse import metrics as M  # noqa: E402
from neuropulse.contract import DEFAULT_SCHEMA  # noqa: E402
from neuropulse.synthetic import CLASSES, make_dataset, noise_columns, band_power_columns  # noqa: E402

warnings.filterwarnings("ignore")
COLUMNS = list(DEFAULT_SCHEMA.columns)
DEFAULT_SEPARATION = 4.0
SWEEP = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
N_DESC, N_CH = 7, 4

# Fixed permutations (documented in the manuscript).
DESCRIPTOR_PERMS = {
    "Descriptors reversed": [6, 5, 4, 3, 2, 1, 0],
    "Descriptors shifted": [1, 2, 3, 4, 5, 6, 0],
    "Descriptors mixed": [int(i) for i in np.random.default_rng(7).permutation(7)],
}
CHANNEL_PERMS = {
    "Channels reversed": [3, 2, 1, 0],
    "Channels shifted": [1, 2, 3, 0],
    "Channels mixed": [2, 0, 3, 1],
}


def column_index(desc_perm=None, ch_perm=None):
    """Index vector that reorders the 28 channel-major columns."""
    desc_perm = list(range(N_DESC)) if desc_perm is None else desc_perm
    ch_perm = list(range(N_CH)) if ch_perm is None else ch_perm
    return np.array([c * N_DESC + d for c in ch_perm for d in desc_perm])


def make_models(seed):
    sc = StandardScaler
    return {
        "RBF-SVM": make_pipeline(sc(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "Logistic regression": make_pipeline(sc(), LogisticRegression(max_iter=2000)),
        "k-NN (k = 5)": make_pipeline(sc(), KNeighborsClassifier(n_neighbors=5)),
        "MLP": make_pipeline(sc(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                                 max_iter=300, random_state=seed)),
        "Random forest": make_pipeline(sc(), RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        "Gradient boosting": make_pipeline(sc(), HistGradientBoostingClassifier(random_state=seed)),
    }


def replicate(r, separation=DEFAULT_SEPARATION):
    df = make_dataset(n_per_class=500, separation=separation, seed=1000 + r)
    X, y = df[COLUMNS].to_numpy(), df["label"].to_numpy()
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.30, stratify=y, random_state=r)
    Xval, Xte, yval, yte = train_test_split(Xrest, yrest, test_size=0.50, stratify=yrest, random_state=r)
    return Xtr, ytr, Xte, yte


def score(model, X, y):
    pred = model.predict(X)
    return float(np.mean(pred == y)), float(f1_score(y, pred, average="macro")), pred


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def e1_baselines(reps):
    out = {m: {"acc": [], "f1": [], "angry_fear_error_share": []} for m in make_models(0)}
    pooled = {m: {"y": [], "pred": []} for m in out}
    for r in range(reps):
        Xtr, ytr, Xte, yte = replicate(r)
        for name, model in make_models(r).items():
            model.fit(Xtr, ytr)
            acc, f1, pred = score(model, Xte, yte)
            wrong = pred != yte
            af = np.isin(yte[wrong], ["ANGRY", "FEAR"]) & np.isin(pred[wrong], ["ANGRY", "FEAR"])
            out[name]["acc"].append(100 * acc)
            out[name]["f1"].append(f1)
            out[name]["angry_fear_error_share"].append(float(af.mean()) if wrong.any() else float("nan"))
            pooled[name]["y"].extend(yte.tolist())
            pooled[name]["pred"].extend(pred.tolist())
    res = {}
    for name, d in out.items():
        n_ok = int(sum(np.array(pooled[name]["y"]) == np.array(pooled[name]["pred"])))
        n = len(pooled[name]["y"])
        lo, hi = M.clopper_pearson(n_ok, n)
        res[name] = {"acc": mean_sd(d["acc"]), "macro_f1": mean_sd(d["f1"]),
                     "angry_fear_error_share": float(np.nanmean(d["angry_fear_error_share"])),
                     "pooled_correct": n_ok, "pooled_n": n, "pooled_ci95": [100 * lo, 100 * hi]}
    ref = np.array(pooled["RBF-SVM"]["pred"])
    y = np.array(pooled["RBF-SVM"]["y"])
    for name in out:
        if name != "RBF-SVM":
            b, c, p = M.mcnemar_exact(y, ref, np.array(pooled[name]["pred"]))
            res[name]["mcnemar_vs_svm"] = {"svm_only_correct": b, "other_only_correct": c, "p": p}
    return res


def e2_sweep(reps):
    keep = ("RBF-SVM", "Logistic regression", "MLP", "Random forest", "Gradient boosting")
    res = {m: {} for m in keep}
    for s in SWEEP:
        acc = {m: [] for m in keep}
        for r in range(reps):
            Xtr, ytr, Xte, yte = replicate(r, separation=s)
            for name, model in make_models(r).items():
                if name in keep:
                    model.fit(Xtr, ytr)
                    acc[name].append(100 * score(model, Xte, yte)[0])
        for m in keep:
            res[m][str(s)] = mean_sd(acc[m])
    return res


def e3_controls(reps):
    keep = ("RBF-SVM", "MLP", "Random forest")
    cond = ["All 28 features", "Band-power features only (20)", "Label-independent features only (8)",
            "All features, shuffled training labels"]
    res = {m: {c: [] for c in cond} for m in keep}
    ix_band = [COLUMNS.index(c) for c in band_power_columns()]
    ix_noise = [COLUMNS.index(c) for c in noise_columns()]
    for r in range(reps):
        Xtr, ytr, Xte, yte = replicate(r)
        rng = np.random.default_rng(r)
        for name, model in make_models(r).items():
            if name not in keep:
                continue
            for c, cols, labels in ((cond[0], slice(None), ytr), (cond[1], ix_band, ytr), (cond[2], ix_noise, ytr),
                                    (cond[3], slice(None), rng.permutation(ytr))):
                fresh = make_models(r)[name]
                fresh.fit(Xtr[:, cols], labels)
                res[name][c].append(100 * score(fresh, Xte[:, cols], yte)[0])
    return {m: {c: mean_sd(v) for c, v in d.items()} for m, d in res.items()}


def e4_order(reps):
    keep = ("RBF-SVM", "MLP", "Random forest", "Gradient boosting")
    perms = {name: column_index(desc_perm=p) for name, p in DESCRIPTOR_PERMS.items()}
    perms.update({name: column_index(ch_perm=p) for name, p in CHANNEL_PERMS.items()})
    res = {m: {p: {"evaluate_only": [], "retrained": []} for p in perms} for m in keep}
    base = {m: [] for m in keep}
    for r in range(reps):
        Xtr, ytr, Xte, yte = replicate(r)
        for name, model in make_models(r).items():
            if name not in keep:
                continue
            model.fit(Xtr, ytr)
            base[name].append(100 * score(model, Xte, yte)[0])
            for p, idx in perms.items():
                res[name][p]["evaluate_only"].append(100 * score(model, Xte[:, idx], yte)[0])
                fresh = make_models(r)[name]
                fresh.fit(Xtr[:, idx], ytr)
                res[name][p]["retrained"].append(100 * score(fresh, Xte[:, idx], yte)[0])
    out = {"unpermuted": {m: mean_sd(v) for m, v in base.items()}, "permutations": {}}
    for m in keep:
        out["permutations"][m] = {p: {k: mean_sd(v) for k, v in d.items()} for p, d in res[m].items()}
    out["definitions"] = {"descriptor": DESCRIPTOR_PERMS, "channel": CHANNEL_PERMS}
    return out


def e5_learning(reps):
    keep = ("RBF-SVM", "Logistic regression", "MLP", "Random forest")
    res = {m: {} for m in keep}
    for f in FRACTIONS:
        acc = {m: [] for m in keep}
        for r in range(reps):
            Xtr, ytr, Xte, yte = replicate(r)
            n = int(round(f * len(ytr)))
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n, stratify=ytr, random_state=r) if f < 1 else (Xtr, None, ytr, None)
            for name, model in make_models(r).items():
                if name in keep:
                    model.fit(Xs, ys)
                    acc[name].append(100 * score(model, Xte, yte)[0])
        for m in keep:
            res[m][str(int(round(f * 1750)))] = mean_sd(acc[m])
    return res


def e6_channel(reps):
    keep = ("RBF-SVM", "MLP", "Random forest")
    names = DEFAULT_SCHEMA.channels
    res = {m: {n: [] for n in names} for m in keep}
    base = {m: [] for m in keep}
    for r in range(reps):
        Xtr, ytr, Xte, yte = replicate(r)
        train_mean = Xtr.mean(axis=0)
        for name, model in make_models(r).items():
            if name not in keep:
                continue
            model.fit(Xtr, ytr)
            base[name].append(100 * score(model, Xte, yte)[0])
            for c, ch in enumerate(names):
                Xn = Xte.copy()
                cols = slice(c * N_DESC, (c + 1) * N_DESC)
                Xn[:, cols] = train_mean[cols]
                res[name][ch].append(100 * score(model, Xn, yte)[0])
    return {"unmodified": {m: mean_sd(v) for m, v in base.items()},
            "neutralised": {m: {ch: mean_sd(v) for ch, v in d.items()} for m, d in res.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="2 replicates, for a smoke test (does not write the JSON)")
    args = ap.parse_args()
    reps = 2 if args.quick else args.replicates
    t0 = time.time()
    results = {
        "meta": {"replicates": reps, "default_separation": DEFAULT_SEPARATION, "sweep": list(SWEEP),
                 "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                 "n_per_class": 500, "split": "70/15/15 stratified; validation partition unused (no tuning)",
                 "data_seed_rule": "1000 + replicate index", "split_seed_rule": "replicate index"},
    }
    for key, fn in (("E1_baselines", e1_baselines), ("E2_sweep", e2_sweep), ("E3_controls", e3_controls),
                    ("E4_order", e4_order), ("E5_learning_curve", e5_learning), ("E6_channel_neutralisation", e6_channel)):
        results[key] = fn(reps)
        print(f"{key} done ({time.time() - t0:.0f} s)")
    if not args.quick:
        path = ROOT / "docs" / "supplementary_results.json"
        path.write_text(json.dumps(results, indent=1))
        print("wrote", path)


if __name__ == "__main__":
    main()
