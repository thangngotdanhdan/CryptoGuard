"""Small, dependency-light chronological split helpers.

These helpers only assign time-step groups; they do NOT prevent graph- or
feature-level temporal leakage. See docs/EXPERIMENT_PROTOCOL.md.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def build_temporal_split(
    time_steps: Iterable[int],
    *,
    train_end: int = 29,
    val_end: int = 34,
    test_end: int = 49,
) -> TemporalSplit:
    """Return disjoint masks for time steps 1..test_end.

    Args:
        time_steps: One time step per node, using the 1-based Elliptic steps.
        train_end: Last training time step (inclusive).
        val_end: Last validation time step (inclusive).
        test_end: Last testing time step (inclusive).

    Raises:
        ValueError: If boundaries are invalid or the input has out-of-range steps.
    """
    if not (1 <= train_end < val_end < test_end):
        raise ValueError("Expected 1 <= train_end < val_end < test_end")

    times = np.asarray(list(time_steps))
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("time_steps must be a non-empty one-dimensional sequence")
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("time_steps must contain integers")
    if np.any((times < 1) | (times > test_end)):
        raise ValueError("time_steps contain values outside the requested time window")

    train = times <= train_end
    val = (times > train_end) & (times <= val_end)
    test = (times > val_end) & (times <= test_end)
    if not np.all(train.astype(int) + val.astype(int) + test.astype(int) == 1):
        raise AssertionError("Every node must be assigned to exactly one time split")
    return TemporalSplit(train=train, val=val, test=test)
