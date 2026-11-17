"""Experimental results reported in the manuscript.

These values are the single source of truth for the figures, the derived
tables and the consistency checks. They were produced by the authors'
experiment notebook on the synthetic dataset described in Section 3 of the
manuscript (2,500 records, seed 42). Nothing in this module is re-estimated.
"""

CLASSES = ("ANGRY", "CALM", "FEAR", "HAPPY", "SAD")
N_TEST = 375
N_PER_CLASS_TEST = 75
N_TRAIN = 1750
N_VALIDATION = 375
N_TOTAL = 2500

MODELS = ("CNN", "CNN-attention", "MLP", "RBF-SVM")

# Off-diagonal test errors per model for the single run.
# Every model confused only ANGRY and FEAR; all other cells are diagonal.
# Tuple: (ANGRY predicted as FEAR, FEAR predicted as ANGRY).
ERRORS = {
    "CNN": (6, 8),
    "CNN-attention": (5, 4),
    "MLP": (10, 5),
    "RBF-SVM": (7, 1),
}

# Single-run accuracy (%) as printed in the manuscript.
SINGLE_RUN_ACCURACY = {
    "CNN": 96.267,
    "CNN-attention": 97.600,
    "MLP": 96.000,
    "RBF-SVM": 97.867,
}

# Three-seed summary over seeds 42, 123 and 2024 (sample standard deviation).
SEEDS = (42, 123, 2024)
THREE_SEED = {
    "CNN": {"acc_mean": 96.800, "acc_sd": 0.462, "f1_mean": 0.96797, "f1_sd": 0.00465},
    "CNN-attention": {"acc_mean": 97.333, "acc_sd": 0.267, "f1_mean": 0.97331, "f1_sd": 0.00264},
    "MLP": {"acc_mean": 96.089, "acc_sd": 0.407, "f1_mean": 0.96086, "f1_sd": 0.00406},
    "RBF-SVM": {"acc_mean": 97.867, "acc_sd": 0.000, "f1_mean": 0.97863, "f1_sd": 0.00000},
}

# CNN-attention, single run.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_ACCURACY_CI = (96.00, 98.93)
BOOTSTRAP_MACRO_F1_CI = (0.95996, 0.98933)
CALIBRATION = {"ece_10bin": 0.0212, "brier_multiclass": 0.0422, "mean_confidence": 0.9959}
REPORTED_MACRO = {"precision": 0.9760, "recall": 0.9760, "f1": 0.9760, "balanced_accuracy": 0.9760}
REPORTED_CNN_ATT_PER_CLASS = {  # precision, recall, F1
    "ANGRY": (0.9459, 0.9333, 0.9396),
    "CALM": (1.0, 1.0, 1.0),
    "FEAR": (0.9342, 0.9467, 0.9404),
    "HAPPY": (1.0, 1.0, 1.0),
    "SAD": (1.0, 1.0, 1.0),
}

# Feature-order perturbation of the trained CNN-attention model (accuracy, %).
ORDER_SENSITIVITY = {
    "Unpermuted": 97.60,
    "Channel order A": 97.33,
    "Channel order B": 97.07,
    "Channel order C": 97.07,
    "Descriptors reversed": 28.00,
    "Descriptors shifted": 21.60,
    "Descriptors mixed": 1.33,
}

# Accuracy (%) after removing one channel row (CNN-attention).
CHANNEL_ABLATION = {"No TP9": 93.60, "No AF7": 95.733, "No AF8": 94.40, "No TP10": 95.20}

# Model-size bookkeeping stated in the manuscript.
CNN_ATTENTION_PARAMETERS = 42821
