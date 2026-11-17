"""Input contract of the NeuroPulse dashboard.

This module re-implements, without Streamlit or Keras, the behaviour that
Section 4.1 and Section 7 of the manuscript describe: selection of the 28
named columns in canonical order, numeric conversion, masking of non-finite
rows, reuse of a training-fitted scaler, reshaping to a 4 x 7 tensor,
softmax-based confidence and batch summaries. It exists so that the contract
can be tested in isolation and so that other front ends can reuse it.

The default column names follow the pattern ``<channel>_<descriptor>``
(for example ``TP9_delta``). If a deployment uses a different header, create
a :class:`FeatureSchema` with the appropriate ``separator`` or subclass
:meth:`FeatureSchema.column_name`.
"""
from __future__ import annotations

import csv
import io
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- errors
class ContractError(ValueError):
    """Base class for all controlled input-contract failures."""


class ParseError(ContractError):
    """The uploaded table could not be parsed."""


class SchemaError(ContractError):
    """Required columns are missing or ambiguous."""


class EmptyInputError(ContractError):
    """No rows, or no valid rows, are available for prediction."""


class RowIndexError(ContractError):
    """A requested row index is outside the table."""


class ArtifactError(ContractError):
    """Saved artifacts are missing or incompatible with the schema."""


# -------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class FeatureSchema:
    """Canonical feature order: channel-major, descriptors within channel."""

    channels: Tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10")
    descriptors: Tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma", "mean", "std")
    separator: str = "_"

    def column_name(self, channel: str, descriptor: str) -> str:
        return f"{channel}{self.separator}{descriptor}"

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(self.column_name(c, d) for c in self.channels for d in self.descriptors)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.channels), len(self.descriptors)

    @property
    def n_features(self) -> int:
        return len(self.channels) * len(self.descriptors)


DEFAULT_SCHEMA = FeatureSchema()
DEFAULT_CLASSES = ("ANGRY", "CALM", "FEAR", "HAPPY", "SAD")


# ------------------------------------------------------------------------- parsing
def read_table_from_text(text: str) -> pd.DataFrame:
    """Parse CSV text, rejecting empty input and ambiguous (duplicate) headers.

    pandas silently renames duplicate headers (``x``, ``x.1``), which would hide
    an ambiguous schema, so the header row is inspected before parsing.
    """
    if text is None or not text.strip():
        raise EmptyInputError("The uploaded table is empty.")
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (csv.Error, StopIteration) as exc:
        raise ParseError(f"The header row could not be read: {exc}") from exc
    names = [h.strip() for h in header]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SchemaError("Duplicate column names make the schema ambiguous: " + ", ".join(duplicates))
    try:
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise ParseError(f"The table could not be parsed: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_table(path: "str | Path") -> pd.DataFrame:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ParseError(f"The file could not be read: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError(f"The file is not valid UTF-8 text: {exc}") from exc
    return read_table_from_text(text)


# ------------------------------------------------------------------- validation
def validate_columns(columns: Sequence[str], schema: FeatureSchema = DEFAULT_SCHEMA) -> None:
    cols = [str(c).strip() for c in columns]
    required = schema.columns
    missing = [c for c in required if c not in cols]
    if missing:
        raise SchemaError(f"{len(missing)} required column(s) missing: " + ", ".join(missing))
    duplicated = [c for c in required if cols.count(c) > 1]
    if duplicated:
        raise SchemaError("Duplicate required columns: " + ", ".join(duplicated))


def validate_row_index(index: object, n_rows: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise RowIndexError("The row index must be an integer.")
    if not 0 <= int(index) < n_rows:
        raise RowIndexError(f"Row index {index} is outside the table (0 to {n_rows - 1}).")
    return int(index)


def to_numeric_matrix(df: pd.DataFrame, schema: FeatureSchema = DEFAULT_SCHEMA) -> Tuple[np.ndarray, np.ndarray]:
    """Return the canonical feature matrix and a validity mask.

    Columns are selected by name, so the order of the uploaded columns is
    irrelevant. Non-numeric entries become NaN. A row is valid only when all 28
    converted values are finite.
    """
    validate_columns(df.columns, schema)
    values = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in schema.columns]
    )
    valid = np.isfinite(values).all(axis=1)
    return values, valid


# --------------------------------------------------------------------- artifacts
def check_artifacts(scaler: object, class_names: Optional[Sequence[str]], schema: FeatureSchema = DEFAULT_SCHEMA,
                    expected_classes: Sequence[str] = DEFAULT_CLASSES) -> None:
    """Reject missing or mismatched artifacts.

    A scaler fitted on a different feature count or order, or an encoder with a
    different class order, would still run and return plausible but wrong
    output, so both are checked explicitly.
    """
    if scaler is None or class_names is None:
        raise ArtifactError("A required artifact (scaler or label encoder) is not available.")
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is not None and int(n_in) != schema.n_features:
        raise ArtifactError(f"The scaler expects {n_in} features, but the schema defines {schema.n_features}.")
    names_in = getattr(scaler, "feature_names_in_", None)
    if names_in is not None and list(names_in) != list(schema.columns):
        raise ArtifactError("The scaler was fitted with a different feature order than the canonical schema.")
    if tuple(class_names) != tuple(expected_classes):
        raise ArtifactError(f"The class order {tuple(class_names)} differs from the expected {tuple(expected_classes)}.")


# ---------------------------------------------------------------------- inference
def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def scale_and_reshape(values: np.ndarray, scaler: object, schema: FeatureSchema = DEFAULT_SCHEMA) -> np.ndarray:
    """Apply the training-fitted scaler and reshape to (n, channels, descriptors)."""
    # Pass named columns when the scaler was fitted with names, so that scikit-learn can
    # verify the order itself instead of warning about unnamed input.
    data = pd.DataFrame(values, columns=schema.columns) if hasattr(scaler, "feature_names_in_") else values
    scaled = np.asarray(scaler.transform(data), dtype=float)
    return scaled.reshape((-1,) + schema.shape)


@dataclass
class BatchResult:
    predictions: pd.DataFrame
    n_total: int
    n_valid: int
    n_invalid: int
    invalid_rows: List[int] = field(default_factory=list)


def predict_batch(df: Optional[pd.DataFrame], scaler: object, predict_proba: Callable[[np.ndarray], np.ndarray],
                  class_names: Sequence[str] = DEFAULT_CLASSES, schema: FeatureSchema = DEFAULT_SCHEMA,
                  expected_classes: Sequence[str] = DEFAULT_CLASSES) -> BatchResult:
    """Predict for every valid row and report how many rows were excluded."""
    if df is None or len(df) == 0:
        raise EmptyInputError("The table contains no rows.")
    check_artifacts(scaler, class_names, schema, expected_classes=expected_classes)
    values, valid = to_numeric_matrix(df, schema)
    if not valid.any():
        raise EmptyInputError("Every row contains missing, non-numeric or non-finite values.")
    tensor = scale_and_reshape(values[valid], scaler, schema)
    probabilities = np.asarray(predict_proba(tensor), dtype=float)
    if probabilities.shape != (tensor.shape[0], len(class_names)) or not np.isfinite(probabilities).all():
        raise ArtifactError("The model returned scores with an unexpected shape or non-finite values.")
    positions = np.flatnonzero(valid)
    labels = np.asarray(class_names)[probabilities.argmax(axis=1)]
    out = pd.DataFrame(
        {"row": positions, "predicted": labels, "confidence_percent": 100.0 * probabilities.max(axis=1)}
    )
    for k, name in enumerate(class_names):
        out[f"score_{name}"] = probabilities[:, k]
    return BatchResult(out, int(len(df)), int(valid.sum()), int((~valid).sum()), np.flatnonzero(~valid).tolist())


def predict_single(df: pd.DataFrame, row_index: object, scaler: object,
                   predict_proba: Callable[[np.ndarray], np.ndarray],
                   class_names: Sequence[str] = DEFAULT_CLASSES, schema: FeatureSchema = DEFAULT_SCHEMA,
                   expected_classes: Sequence[str] = DEFAULT_CLASSES) -> pd.Series:
    i = validate_row_index(row_index, 0 if df is None else len(df))
    result = predict_batch(df.iloc[[i]], scaler, predict_proba, class_names, schema, expected_classes)
    if result.n_valid == 0:
        raise EmptyInputError(f"Row {i} contains missing, non-numeric or non-finite values.")
    row = result.predictions.iloc[0].copy()
    row["row"] = i
    return row


def summarise(predictions: pd.DataFrame, class_names: Sequence[str] = DEFAULT_CLASSES) -> Dict[str, object]:
    """Batch summary over the M valid predictions (counts, shares, top class, mean confidence)."""
    m = len(predictions)
    if m == 0:
        raise EmptyInputError("No valid predictions to summarise.")
    counts = {k: int((predictions["predicted"] == k).sum()) for k in class_names}
    shares = {k: 100.0 * v / m for k, v in counts.items()}
    top = max(class_names, key=lambda k: (counts[k], -list(class_names).index(k)))
    return {
        "denominator": m,
        "counts": counts,
        "shares_percent": shares,
        "top_class": top,
        "mean_confidence_percent": float(predictions["confidence_percent"].mean()),
    }


# ----------------------------------------------------------------------- history
class HistoryStore:
    """Minimal SQLite history of predictions (timestamp, row, label, confidence)."""

    def __init__(self, path: "str | Path" = "neuropulse_history.db") -> None:
        self.path = str(path)
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, row_number INTEGER NOT NULL, "
                "predicted_emotion TEXT NOT NULL, confidence REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, row_number: int, predicted_emotion: str, confidence: float) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO predictions (timestamp, row_number, predicted_emotion, confidence) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), int(row_number), str(predicted_emotion),
                 float(confidence)),
            )

    def all(self) -> List[Tuple]:
        with self._connect() as con:
            return con.execute(
                "SELECT timestamp, row_number, predicted_emotion, confidence FROM predictions ORDER BY id"
            ).fetchall()
