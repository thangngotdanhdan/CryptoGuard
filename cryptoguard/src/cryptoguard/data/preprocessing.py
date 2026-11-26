"""Leakage-safe preprocessing, feature-set selection, and mask utilities.

This module intentionally depends only on the Python standard library and NumPy
at import time so lightweight CI environments (`.[dev]`) stay green without
requiring pandas, scikit-learn, torch, or torch_geometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from cryptoguard.data.temporal_split import TemporalSplit, build_temporal_split

UNKNOWN_LABEL = -1
LICIT_LABEL = 0
ILLICIT_LABEL = 1

NUM_LOCAL_FEATURES = 93
NUM_TOTAL_FEATURES = 165

FeatureSet = Literal["local_only", "local_plus_aggregated"]
FitScope = Literal["train_time", "train_labeled"]

FEATURE_SET_DIMS: dict[FeatureSet, int] = {
    "local_only": NUM_LOCAL_FEATURES,
    "local_plus_aggregated": NUM_TOTAL_FEATURES,
}


@dataclass(frozen=True)
class LabeledTemporalMasks:
    """Pairs chronological time-window masks with labeled-only evaluation masks."""

    time_split: TemporalSplit
    train_labeled: np.ndarray
    val_labeled: np.ndarray
    test_labeled: np.ndarray

    @property
    def train_time(self) -> np.ndarray:
        return self.time_split.train

    @property
    def val_time(self) -> np.ndarray:
        return self.time_split.val

    @property
    def test_time(self) -> np.ndarray:
        return self.time_split.test


def _as_valid_labels(y: Iterable[int], *, expected_len: int, unknown_label: int) -> np.ndarray:
    labels = np.asarray(list(y)) if not isinstance(y, np.ndarray) else np.asarray(y)
    if labels.ndim != 1 or len(labels) != expected_len:
        raise ValueError("y must be a 1D array with the same length as time_steps")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y must contain integer class labels")

    allowed = np.array([unknown_label, LICIT_LABEL, ILLICIT_LABEL], dtype=labels.dtype)
    if not np.all(np.isin(labels, allowed)):
        raise ValueError(
            f"y contains unsupported label values; expected only {allowed.tolist()}"
        )
    return labels


def _validate_feature_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("x must be a non-empty 2D numeric array")
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ):
        raise ValueError("x must contain numeric values")
    if not np.isfinite(arr).all():
        raise ValueError("x contains NaN or infinite values")
    return arr


def _validate_bool_mask(mask: np.ndarray, *, expected_len: int, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 1 or len(arr) != expected_len:
        raise ValueError(f"{name} must be a 1D boolean array of length {expected_len}")
    if arr.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    return arr


def _validate_edge_endpoints(
    src: np.ndarray, dst: np.ndarray, *, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    src_arr = np.asarray(src)
    dst_arr = np.asarray(dst)
    if src_arr.ndim != 1 or dst_arr.ndim != 1 or len(src_arr) != len(dst_arr):
        raise ValueError("src and dst must be 1D arrays of equal length")
    if not (np.issubdtype(src_arr.dtype, np.integer) and np.issubdtype(dst_arr.dtype, np.integer)):
        raise ValueError("src and dst must contain integer node indices")
    if len(src_arr) > 0:
        if np.any((src_arr < 0) | (src_arr >= num_nodes)) or np.any(
            (dst_arr < 0) | (dst_arr >= num_nodes)
        ):
            raise ValueError("Edge indices are out of bounds for the node array")
    return src_arr, dst_arr


def build_labeled_temporal_masks(
    time_steps: Iterable[int],
    y: Iterable[int],
    *,
    train_end: int = 29,
    val_end: int = 34,
    test_end: int = 49,
    unknown_label: int = UNKNOWN_LABEL,
) -> LabeledTemporalMasks:
    """Build chronological time masks and exclude unknown nodes from labeled masks."""
    time_split = build_temporal_split(
        time_steps,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )
    labels = _as_valid_labels(y, expected_len=len(time_split.train), unknown_label=unknown_label)
    labeled = labels != unknown_label

    return LabeledTemporalMasks(
        time_split=time_split,
        train_labeled=time_split.train & labeled,
        val_labeled=time_split.val & labeled,
        test_labeled=time_split.test & labeled,
    )


def select_feature_set(
    x: np.ndarray,
    feature_set: FeatureSet = "local_plus_aggregated",
) -> np.ndarray:
    """Return a float32 copy containing the requested Elliptic feature subset."""
    if feature_set not in FEATURE_SET_DIMS:
        raise ValueError(
            f"Unsupported feature_set {feature_set!r}; expected one of {list(FEATURE_SET_DIMS)}"
        )

    arr = _validate_feature_matrix(x)
    if arr.shape[1] != NUM_TOTAL_FEATURES:
        raise ValueError(
            f"Expected {NUM_TOTAL_FEATURES} raw features, got shape {arr.shape}"
        )

    num_cols = FEATURE_SET_DIMS[feature_set]
    return np.array(arr[:, :num_cols], dtype=np.float32, copy=True, order="C")


class TrainOnlyStandardScaler:
    """StandardScaler fitted strictly on a boolean training mask."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_samples_seen_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, x: np.ndarray, fit_mask: np.ndarray) -> TrainOnlyStandardScaler:
        """Estimate feature mean and std on `x[fit_mask]` using float64 accumulation."""
        arr = _validate_feature_matrix(x)
        mask = _validate_bool_mask(fit_mask, expected_len=arr.shape[0], name="fit_mask")
        if not np.any(mask):
            raise ValueError("fit_mask must select at least one training sample")

        x_fit = arr[mask].astype(np.float64, copy=False)
        mean = np.mean(x_fit, axis=0)
        std = np.std(x_fit, axis=0, ddof=0)

        # Keep constant training features finite (matching sklearn StandardScaler).
        self.mean_ = mean
        self.scale_ = np.where(std == 0.0, 1.0, std)
        self.n_samples_seen_ = int(x_fit.shape[0])
        self.n_features_in_ = int(arr.shape[1])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize `x` using parameters fitted on training samples only."""
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise ValueError("TrainOnlyStandardScaler must be fitted before calling transform")

        arr = _validate_feature_matrix(x)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {arr.shape[1]}"
            )

        scaled = ((arr.astype(np.float64, copy=False) - self.mean_) / self.scale_).astype(
            np.float32
        )
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled features contain NaN or infinite values")
        return scaled

    def fit_transform(self, x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
        """Fit on `fit_mask` and return standardized features for all rows."""
        return self.fit(x, fit_mask).transform(x)


@dataclass(frozen=True)
class PreprocessedFeatures:
    """Bundled standardized features and the train-fitted scaler that produced them."""

    x: np.ndarray
    feature_set: FeatureSet
    scaler: TrainOnlyStandardScaler
    fit_scope: FitScope


def preprocess_features(
    x: np.ndarray,
    masks: LabeledTemporalMasks,
    *,
    feature_set: FeatureSet = "local_plus_aggregated",
    fit_on: FitScope = "train_time",
) -> PreprocessedFeatures:
    """Slice a feature set and standardize it using only training-window statistics."""
    if fit_on not in ("train_time", "train_labeled"):
        raise ValueError("fit_on must be either 'train_time' or 'train_labeled'")

    x_selected = select_feature_set(x, feature_set)
    expected_nodes = len(masks.train_time)
    if x_selected.shape[0] != expected_nodes:
        raise ValueError(
            f"x has {x_selected.shape[0]} rows, but masks expect {expected_nodes} nodes"
        )

    fit_mask = masks.train_time if fit_on == "train_time" else masks.train_labeled
    scaler = TrainOnlyStandardScaler()
    x_scaled = scaler.fit_transform(x_selected, fit_mask)

    return PreprocessedFeatures(
        x=x_scaled,
        feature_set=feature_set,
        scaler=scaler,
        fit_scope=fit_on,
    )


def prepare_feature_sets(
    x: np.ndarray,
    masks: LabeledTemporalMasks,
    *,
    fit_on: FitScope = "train_time",
) -> dict[FeatureSet, PreprocessedFeatures]:
    """Prepare both `local_only` (93) and `local_plus_aggregated` (165) feature sets."""
    return {
        "local_only": preprocess_features(
            x, masks, feature_set="local_only", fit_on=fit_on
        ),
        "local_plus_aggregated": preprocess_features(
            x, masks, feature_set="local_plus_aggregated", fit_on=fit_on
        ),
    }


def count_cross_timestep_edges(
    src: np.ndarray,
    dst: np.ndarray,
    time_step: np.ndarray,
) -> int:
    """Count directed edges whose endpoints belong to different time steps."""
    times = np.asarray(time_step)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("time_step must be a non-empty 1D array")
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("time_step must contain integers")

    src_arr, dst_arr = _validate_edge_endpoints(src, dst, num_nodes=len(times))
    if len(src_arr) == 0:
        return 0
    return int(np.sum(times[src_arr] != times[dst_arr]))


def filter_edges_by_mask(
    src: np.ndarray,
    dst: np.ndarray,
    node_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed edges whose source and destination both satisfy `node_mask`."""
    mask = np.asarray(node_mask)
    if mask.ndim != 1 or len(mask) == 0 or mask.dtype != np.bool_:
        raise ValueError("node_mask must be a non-empty 1D boolean array")

    src_arr, dst_arr = _validate_edge_endpoints(src, dst, num_nodes=len(mask))
    keep = mask[src_arr] & mask[dst_arr]
    return src_arr[keep].copy(), dst_arr[keep].copy()


def summarize_split(
    y: np.ndarray,
    masks: LabeledTemporalMasks,
) -> dict[str, dict[str, int | float]]:
    """Summarize node and class counts across train, validation, and test windows."""
    labels = _as_valid_labels(
        y, expected_len=len(masks.train_time), unknown_label=UNKNOWN_LABEL
    )
    windows = (
        ("train", masks.train_time),
        ("val", masks.val_time),
        ("test", masks.test_time),
    )

    summary: dict[str, dict[str, int | float]] = {}
    for name, window_mask in windows:
        split_labels = labels[window_mask]
        licit = int(np.sum(split_labels == LICIT_LABEL))
        illicit = int(np.sum(split_labels == ILLICIT_LABEL))
        unknown = int(np.sum(split_labels == UNKNOWN_LABEL))
        labeled = licit + illicit
        summary[name] = {
            "total": int(len(split_labels)),
            "labeled": labeled,
            "unknown": unknown,
            "licit": licit,
            "illicit": illicit,
            "illicit_rate": float(illicit / labeled) if labeled > 0 else 0.0,
        }
    return summary


def attach_masks_to_graph(
    graph,
    masks: LabeledTemporalMasks,
    *,
    x_scaled: np.ndarray | None = None,
    feature_set: str | None = None,
):
    """Clone a PyG Data object and attach labeled masks, time masks, and scaled features."""
    import torch

    new_graph = graph.clone()
    num_nodes = int(new_graph.num_nodes)
    if num_nodes != len(masks.train_time):
        raise ValueError(
            f"Graph has {num_nodes} nodes, but masks expect {len(masks.train_time)}"
        )

    new_graph.train_mask = torch.from_numpy(masks.train_labeled).bool()
    new_graph.val_mask = torch.from_numpy(masks.val_labeled).bool()
    new_graph.test_mask = torch.from_numpy(masks.test_labeled).bool()

    new_graph.train_time_mask = torch.from_numpy(masks.train_time).bool()
    new_graph.val_time_mask = torch.from_numpy(masks.val_time).bool()
    new_graph.test_time_mask = torch.from_numpy(masks.test_time).bool()

    if x_scaled is not None:
        arr = _validate_feature_matrix(x_scaled)
        if arr.shape[0] != num_nodes:
            raise ValueError(
                f"x_scaled has {arr.shape[0]} rows, expected {num_nodes}"
            )
        new_graph.x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).float()

    if feature_set is not None:
        new_graph.feature_set = feature_set

    new_graph.validate(raise_on_error=True)
    return new_graph
