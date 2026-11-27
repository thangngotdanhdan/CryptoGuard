"""Elliptic Bitcoin dataset loader with schema and temporal integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cryptoguard.data.validation import validate_strict_int64

MIN_TIME_STEP = 1
MAX_TIME_STEP = 49
NUM_FEATURES = 165
EXPECTED_FEATURE_COLUMNS = NUM_FEATURES + 2  # txId, time_step, feature_1..feature_165

LABEL_MAPPING = {
    "1": 1,
    "2": 0,
    "unknown": -1,
}


@dataclass
class EllipticData:
    x: np.ndarray
    y: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    time_step: np.ndarray
    tx_ids: np.ndarray
    tx_to_idx: dict[int, int]


def validate_transaction_ids(raw_ids: Any, *, name: str = "txId") -> np.ndarray:
    """Strictly validate transaction IDs into canonical int64 without float64 loss."""
    return validate_strict_int64(
        raw_ids,
        name=name,
        allow_integral_floats=False,
    )


def validate_time_steps(
    raw_steps: Any,
    *,
    min_step: int = MIN_TIME_STEP,
    max_step: int = MAX_TIME_STEP,
) -> np.ndarray:
    """Validate Elliptic time steps before converting them to int64.

    Rejects missing, boolean, complex, datetime/timedelta, non-numeric,
    non-finite, fractional, and out-of-range values. Valid original Elliptic
    time steps are integers in ``[min_step, max_step]`` (default ``1..49``).
    """
    return validate_strict_int64(
        raw_steps,
        name="time_step",
        min_value=min_step,
        max_value=max_step,
        allow_integral_floats=True,
    )


def validate_edge_indices(
    src: Any,
    dst: Any,
    *,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly validate directed edge endpoint arrays against ``[0, num_nodes - 1]``."""
    if isinstance(src, np.ndarray) and src.ndim != 1:
        raise ValueError(
            f"Invalid edge arrays: src shape {src.shape} must be a 1D array."
        )
    if isinstance(dst, np.ndarray) and dst.ndim != 1:
        raise ValueError(
            f"Invalid edge arrays: dst shape {dst.shape} must be a 1D array."
        )

    src_validated = validate_strict_int64(
        src,
        name="edge_index (src)",
        allow_integral_floats=False,
    )
    dst_validated = validate_strict_int64(
        dst,
        name="edge_index (dst)",
        allow_integral_floats=False,
    )

    if len(src_validated) != len(dst_validated):
        raise ValueError(
            f"Invalid edge arrays: src shape {src_validated.shape} and "
            f"dst shape {dst_validated.shape} must be 1D arrays of equal length."
        )

    if len(src_validated) > 0:
        if num_nodes <= 0:
            raise ValueError(
                "Invalid edge_index values: graph has 0 nodes, so no edges are permitted."
            )
        src_oob = (src_validated < 0) | (src_validated >= num_nodes)
        dst_oob = (dst_validated < 0) | (dst_validated >= num_nodes)
        if np.any(src_oob) or np.any(dst_oob):
            bad_src = np.unique(src_validated[src_oob])[:5].tolist()
            bad_dst = np.unique(dst_validated[dst_oob])[:5].tolist()
            raise ValueError(
                f"Invalid edge_index values: node indices must be in [0, {num_nodes - 1}] "
                f"(bad src={bad_src}, bad dst={bad_dst})."
            )

    return src_validated, dst_validated


def validate_temporal_edges(
    src: Any,
    dst: Any,
    time_step: Any,
    *,
    tx_ids: np.ndarray | None = None,
) -> None:
    """Validate edge indices and reject edges crossing different time steps."""
    # Check array dimensionality/length first for clear shape diagnostics.
    if isinstance(src, np.ndarray) and src.ndim != 1:
        raise ValueError(
            f"Invalid edge arrays: src shape {src.shape} must be a 1D array."
        )
    if isinstance(dst, np.ndarray) and dst.ndim != 1:
        raise ValueError(
            f"Invalid edge arrays: dst shape {dst.shape} must be a 1D array."
        )

    validated_time_step = validate_time_steps(time_step)
    src_arr, dst_arr = validate_edge_indices(
        src,
        dst,
        num_nodes=len(validated_time_step),
    )

    if len(src_arr) == 0:
        return

    src_steps = validated_time_step[src_arr]
    dst_steps = validated_time_step[dst_arr]
    cross_time_mask = src_steps != dst_steps

    if np.any(cross_time_mask):
        bad_count = int(np.sum(cross_time_mask))
        first_edge = int(np.flatnonzero(cross_time_mask)[0])
        src_idx = int(src_arr[first_edge])
        dst_idx = int(dst_arr[first_edge])
        src_t = int(validated_time_step[src_idx])
        dst_t = int(validated_time_step[dst_idx])

        if tx_ids is not None:
            detail = (
                f"txId {int(tx_ids[src_idx])} (node {src_idx}, time_step={src_t}) -> "
                f"txId {int(tx_ids[dst_idx])} (node {dst_idx}, time_step={dst_t})"
            )
        else:
            detail = (
                f"node {src_idx} (time_step={src_t}) -> "
                f"node {dst_idx} (time_step={dst_t})"
            )

        raise ValueError(
            f"Cross-time-step edges are not permitted in the original Elliptic dataset: "
            f"found {bad_count} invalid directed edge(s); first example: {detail}."
        )


def load_elliptic(data_dir: str | Path) -> EllipticData:
    """Load and validate the original Elliptic Bitcoin dataset from CSV files.

    Transaction IDs and time steps are parsed as raw strings before strict
    canonical int64 validation so values never pass through lossy float64
    conversion (preserving IDs > 2^53 and rejecting tiny fractional tokens).
    """
    data_dir = Path(data_dir)

    # 1. Load raw CSV files with string dtype on ID and time_step fields
    features = pd.read_csv(
        data_dir / "elliptic_txs_features.csv",
        header=None,
        dtype={0: str, 1: str},
    )

    classes = pd.read_csv(
        data_dir / "elliptic_txs_classes.csv",
        dtype={"txId": str, "class": str},
    )

    edges = pd.read_csv(
        data_dir / "elliptic_txs_edgelist.csv",
        dtype={"txId1": str, "txId2": str},
    )

    # 2. Validate feature dimensions
    if features.shape[1] != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("Invalid feature dimensions")

    # 3. Name feature columns
    feature_cols = [f"feature_{i}" for i in range(1, NUM_FEATURES + 1)]
    features.columns = ["txId", "time_step", *feature_cols]

    # 4. Strictly validate transaction IDs across all three CSVs before merging or mapping
    feature_tx_ids = validate_transaction_ids(features["txId"], name="features txId")
    class_tx_ids = validate_transaction_ids(classes["txId"], name="classes txId")
    edge_src_tx_ids = validate_transaction_ids(edges["txId1"], name="edgelist txId1")
    edge_dst_tx_ids = validate_transaction_ids(edges["txId2"], name="edgelist txId2")

    # 5. Strictly validate time steps before merging or integer casting
    time_step = validate_time_steps(features["time_step"])

    # 6. Merge canonical int64 transaction IDs with one-to-one uniqueness enforcement
    features_keyed = pd.DataFrame({"txId": feature_tx_ids})
    classes_keyed = pd.DataFrame({"txId": class_tx_ids, "class": classes["class"]})
    merged_labels = features_keyed.merge(
        classes_keyed,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    if merged_labels["class"].isna().any():
        raise ValueError("Missing transaction labels")

    # 7. Encode labels
    labels_mapped = merged_labels["class"].map(LABEL_MAPPING)
    if labels_mapped.isna().any():
        raise ValueError("Invalid class label")

    # 8. Build transaction ID -> node index mapping (exact int64 keys)
    tx_ids = feature_tx_ids
    tx_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(tx_ids)}

    # 9. Map directed edge endpoints without float64 coercion
    src_list: list[int] = []
    dst_list: list[int] = []
    for src_tx, dst_tx in zip(edge_src_tx_ids.tolist(), edge_dst_tx_ids.tolist(), strict=True):
        src_idx = tx_to_idx.get(src_tx)
        dst_idx = tx_to_idx.get(dst_tx)
        if src_idx is None or dst_idx is None:
            raise ValueError("Invalid edge endpoints")
        src_list.append(src_idx)
        dst_list.append(dst_idx)

    src = np.array(src_list, dtype=np.int64, copy=True, order="C")
    dst = np.array(dst_list, dtype=np.int64, copy=True, order="C")

    # 10. Validate temporal edge integrity (preserves directed graph structure)
    validate_temporal_edges(src, dst, time_step, tx_ids=tx_ids)

    # 11. Convert numerical features and labels to owned, writable NumPy arrays
    x = np.array(
        features[feature_cols].to_numpy(copy=True),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    y = np.array(labels_mapped.to_numpy(copy=True), dtype=np.int64, copy=True, order="C")

    # 12. Validate numerical features
    if not np.isfinite(x).all():
        raise ValueError("Non-finite feature values detected")

    # 13. Return processed data
    return EllipticData(
        x=x,
        y=y,
        src=src,
        dst=dst,
        time_step=time_step,
        tx_ids=tx_ids,
        tx_to_idx=tx_to_idx,
    )
