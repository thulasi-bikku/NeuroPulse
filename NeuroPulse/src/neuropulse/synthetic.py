"""Reference generator for the supplementary experiments.

This generator is released with the paper so that the supplementary
experiments can be repeated end to end. It is *not* the generator that
produced the dataset of the main experiment (file checksum 0cf78566...); its
class profiles were specified independently, and results obtained with it are
reported separately and must not be mixed with the main results.

Design
------
A record has four channel groups (TP9, AF7, AF8, TP10) with seven descriptors
each: delta, theta, alpha, beta, gamma, mean and standard deviation. The 20
band-power descriptors depend on the class through two heuristic latent
dimensions, arousal and valence, plus a small class-specific offset that is the
only source of difference between ANGRY and FEAR. The eight mean and standard
deviation descriptors are independent of the class. All values are in arbitrary
units and band powers are clipped at 0.01.

    x_band = base_band + s * (arousal_k * A + valence_k * V + offset_k) + noise
    noise ~ N(0, noise_sd^2) independently per feature

``separation`` (s) scales the class-dependent term and is the difficulty
parameter of the sweep experiment.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .contract import DEFAULT_SCHEMA, FeatureSchema

CLASSES = ("ANGRY", "CALM", "FEAR", "HAPPY", "SAD")
N_BANDS = 5
BAND_FLOOR = 0.01

# Latent coordinates of each class (heuristic valence-arousal placement).
AROUSAL = {"ANGRY": 1.0, "CALM": -1.0, "FEAR": 1.0, "HAPPY": 0.5, "SAD": -0.5}
VALENCE = {"ANGRY": -1.0, "CALM": 0.5, "FEAR": -0.6, "HAPPY": 1.0, "SAD": -1.0}

BASE = np.array([4.0, 3.0, 2.5, 1.8, 1.2])  # delta, theta, alpha, beta, gamma

# Loadings of the latent dimensions on (channel, band); channels: TP9, AF7, AF8, TP10.
AROUSAL_LOADING = np.zeros((4, N_BANDS))
AROUSAL_LOADING[:, 2] = -0.7   # alpha decreases with arousal
AROUSAL_LOADING[:, 3] = 0.8    # beta increases
AROUSAL_LOADING[:, 4] = 0.9    # gamma increases

VALENCE_LOADING = np.zeros((4, N_BANDS))
VALENCE_LOADING[1, 2], VALENCE_LOADING[2, 2] = -0.6, 0.6   # frontal alpha asymmetry (AF7 vs AF8)
VALENCE_LOADING[0, 1], VALENCE_LOADING[3, 1] = 0.4, 0.4    # temporal theta

# Small class-specific offsets; ANGRY and FEAR differ only here.
OFFSET = {k: np.zeros((4, N_BANDS)) for k in CLASSES}
OFFSET["ANGRY"][1, 4] = OFFSET["ANGRY"][2, 4] = 0.35       # frontal gamma
OFFSET["FEAR"][0, 0] = OFFSET["FEAR"][3, 0] = 0.35         # temporal delta
OFFSET["CALM"][:, 0] = 0.3
OFFSET["SAD"][:, 1] = 0.3
OFFSET["HAPPY"][1:3, 3] = 0.3


def class_means(label: str, separation: float = 1.0) -> np.ndarray:
    """Mean band-power matrix (4 channels x 5 bands) for one class."""
    effect = AROUSAL[label] * AROUSAL_LOADING + VALENCE[label] * VALENCE_LOADING + OFFSET[label]
    return BASE[None, :] + separation * effect


def make_dataset(n_per_class: int = 500, separation: float = 1.0, noise_sd: float = 1.0, seed: int = 0,
                 schema: FeatureSchema = DEFAULT_SCHEMA) -> pd.DataFrame:
    """Return a shuffled DataFrame with the 28 schema columns and a ``label`` column."""
    rng = np.random.default_rng(seed)
    n_desc = len(schema.descriptors)
    rows, labels = [], []
    for label in CLASSES:
        mu = class_means(label, separation)
        x = np.empty((n_per_class, len(schema.channels), n_desc))
        x[:, :, :N_BANDS] = np.maximum(rng.normal(mu[None, :, :], noise_sd, size=(n_per_class, 4, N_BANDS)),
                                       BAND_FLOOR)
        x[:, :, N_BANDS] = rng.normal(0.0, 1.0, size=(n_per_class, 4))            # mean: label-independent
        x[:, :, N_BANDS + 1] = np.abs(rng.normal(1.0, 0.3, size=(n_per_class, 4)))  # std: label-independent
        rows.append(x.reshape(n_per_class, -1))
        labels += [label] * n_per_class
    data = np.vstack(rows)
    df = pd.DataFrame(data, columns=list(schema.columns))
    df["label"] = labels
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def band_power_columns(schema: FeatureSchema = DEFAULT_SCHEMA) -> Tuple[str, ...]:
    return tuple(schema.column_name(c, d) for c in schema.channels for d in schema.descriptors[:N_BANDS])


def noise_columns(schema: FeatureSchema = DEFAULT_SCHEMA) -> Tuple[str, ...]:
    return tuple(schema.column_name(c, d) for c in schema.channels for d in schema.descriptors[N_BANDS:])
