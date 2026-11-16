import numpy as np
import pytest

from cryptoguard.data.temporal_split import build_temporal_split


def test_expected_boundaries():
    split = build_temporal_split([1, 29, 30, 34, 35, 49])
    assert split.train.tolist() == [True, True, False, False, False, False]
    assert split.val.tolist() == [False, False, True, True, False, False]
    assert split.test.tolist() == [False, False, False, False, True, True]


def test_masks_cover_all_nodes_exactly_once():
    split = build_temporal_split(range(1, 50))
    membership = split.train.astype(int) + split.val.astype(int) + split.test.astype(int)
    assert np.all(membership == 1)


@pytest.mark.parametrize("steps", [[0, 1], [1, 50], [1.5, 2], [], ["1", "2"]])
def test_invalid_steps_are_rejected(steps):
    with pytest.raises(ValueError):
        build_temporal_split(steps)


def test_invalid_boundaries_rejected():
    with pytest.raises(ValueError):
        build_temporal_split([1, 2, 3], train_end=34, val_end=29)
