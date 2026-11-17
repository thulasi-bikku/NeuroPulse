"""Model definitions and analytic parameter accounting.

``cnn_attention_layers`` lists the layers of the CNN-attention model with their
output shapes and trainable-parameter counts, computed from the layer
definitions alone. The total (42,821) equals the figure reported by the
experiment notebook, which is possible only when the attention step has no
trainable projections and no learned scale: query, key and value are the
unprojected representation itself.

``build_cnn_attention`` constructs the same network in Keras. TensorFlow is
imported lazily so that the rest of the package works without it.
"""
from __future__ import annotations

from typing import List, Tuple

INPUT_STEPS = 4          # channel groups, treated as the sequence axis
INPUT_FEATURES = 7       # descriptors, treated as the Conv1D input channels
N_CLASSES = 5
KERNEL_SIZE = 2


def _conv1d_params(kernel: int, c_in: int, filters: int) -> int:
    return kernel * c_in * filters + filters


def _dense_params(n_in: int, n_out: int) -> int:
    return n_in * n_out + n_out


def cnn_attention_layers() -> List[Tuple[str, str, int]]:
    """Return (layer, output shape, trainable parameters) for the CNN-attention model."""
    t = INPUT_STEPS
    return [
        ("Input", f"({t}, {INPUT_FEATURES})", 0),
        ("Conv1D, 64 filters, kernel 2, same padding, ReLU", f"({t}, 64)", _conv1d_params(KERNEL_SIZE, INPUT_FEATURES, 64)),
        ("Conv1D, 128 filters, kernel 2, same padding, ReLU", f"({t}, 128)", _conv1d_params(KERNEL_SIZE, 64, 128)),
        ("Self-attention, Q = K = V (no projections)", f"({t}, 128)", 0),
        ("Residual addition", f"({t}, 128)", 0),
        ("Layer normalization", f"({t}, 128)", 2 * 128),
        ("Global average pooling", "(128)", 0),
        ("Dense 128, ReLU", "(128)", _dense_params(128, 128)),
        ("Dropout 0.4", "(128)", 0),
        ("Dense 64, ReLU", "(64)", _dense_params(128, 64)),
        ("Dense 5, softmax", f"({N_CLASSES})", _dense_params(64, N_CLASSES)),
    ]


def cnn_attention_parameter_count() -> int:
    return sum(p for _, _, p in cnn_attention_layers())


def mlp_parameter_count(n_inputs: int = 28) -> int:
    """Dense 128, Dense 64 and a five-class softmax layer on the flattened features."""
    return _dense_params(n_inputs, 128) + _dense_params(128, 64) + _dense_params(64, N_CLASSES)


def attention_matrix_shape() -> Tuple[int, int]:
    """The attention scores relate the four channel-group tokens to each other."""
    return INPUT_STEPS, INPUT_STEPS


def build_cnn_attention():  # pragma: no cover - requires TensorFlow
    """Keras implementation of the CNN-attention model of Figure 2 in the manuscript."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(INPUT_STEPS, INPUT_FEATURES))
    x = layers.Conv1D(64, KERNEL_SIZE, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(128, KERNEL_SIZE, padding="same", activation="relu")(x)
    attended = layers.Attention()([x, x])          # default settings: no learned scale, no projections
    x = layers.Add()([x, attended])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="neuropulse_cnn_attention")


def build_rbf_svm():
    """Standardization followed by an RBF-kernel SVM (C = 1, gamma = 'scale')."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
