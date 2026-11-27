import numpy as np
import pytest

from cryptoguard.data import (
    FEATURE_SET_DIMS,
    ILLICIT_LABEL,
    LICIT_LABEL,
    NUM_LOCAL_FEATURES,
    NUM_TOTAL_FEATURES,
    UNKNOWN_LABEL,
    TrainOnlyStandardScaler,
    build_labeled_temporal_masks,
    count_cross_timestep_edges,
    filter_edges_by_mask,
    prepare_feature_sets,
    preprocess_features,
    select_feature_set,
    summarize_split,
)


def _make_synthetic_dataset(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time_steps = np.array([1, 10, 29, 29, 30, 34, 34, 35, 49, 49], dtype=np.int64)
    y = np.array(
        [
            ILLICIT_LABEL,
            UNKNOWN_LABEL,
            LICIT_LABEL,
            LICIT_LABEL,
            ILLICIT_LABEL,
            UNKNOWN_LABEL,
            LICIT_LABEL,
            LICIT_LABEL,
            UNKNOWN_LABEL,
            ILLICIT_LABEL,
        ],
        dtype=np.int64,
    )
    x = rng.normal(loc=2.0, scale=3.0, size=(len(time_steps), NUM_TOTAL_FEATURES)).astype(
        np.float32
    )
    return time_steps, y, x


def test_labeled_masks_exclude_unknown_and_do_not_overlap():
    time_steps, y, _ = _make_synthetic_dataset()
    masks = build_labeled_temporal_masks(time_steps, y)

    assert masks.train_time.tolist() == [True] * 4 + [False] * 6
    assert masks.val_time.tolist() == [False] * 4 + [True] * 3 + [False] * 3
    assert masks.test_time.tolist() == [False] * 7 + [True] * 3

    assert masks.train_labeled.tolist() == [True, False, True, True] + [False] * 6
    assert masks.val_labeled.tolist() == [False] * 4 + [True, False, True] + [False] * 3
    assert masks.test_labeled.tolist() == [False] * 7 + [True, False, True]

    labeled_membership = (
        masks.train_labeled.astype(int)
        + masks.val_labeled.astype(int)
        + masks.test_labeled.astype(int)
    )
    assert np.all(labeled_membership <= 1)
    assert np.all(y[masks.train_labeled] != UNKNOWN_LABEL)
    assert np.all(y[masks.val_labeled] != UNKNOWN_LABEL)
    assert np.all(y[masks.test_labeled] != UNKNOWN_LABEL)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([0, 1, 2, 0]),
        np.array([0.0, 1.0, -1.0, 0.0]),
        np.array([0, 1]),
    ],
)
def test_invalid_labels_are_rejected(bad_y):
    time_steps = np.array([1, 2, 30, 35], dtype=np.int64)
    with pytest.raises(ValueError):
        build_labeled_temporal_masks(time_steps, bad_y)


def test_select_feature_set_shapes_and_immutability():
    _, _, x = _make_synthetic_dataset()
    x_orig = x.copy()

    local = select_feature_set(x, "local_only")
    full = select_feature_set(x, "local_plus_aggregated")

    assert local.shape == (len(x), FEATURE_SET_DIMS["local_only"])
    assert full.shape == (len(x), FEATURE_SET_DIMS["local_plus_aggregated"])
    assert local.dtype == np.float32
    assert full.dtype == np.float32
    assert np.array_equal(local, x[:, :NUM_LOCAL_FEATURES])
    assert np.array_equal(full, x)

    local[0, 0] = -999.0
    full[0, 0] = -999.0
    assert np.array_equal(x, x_orig)


def test_select_feature_set_rejects_bad_inputs():
    _, _, x = _make_synthetic_dataset()
    with pytest.raises(ValueError):
        select_feature_set(x, "unknown_set")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        select_feature_set(x[:, :10], "local_only")

    x_nan = x.copy()
    x_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        select_feature_set(x_nan, "local_only")


def test_scaler_fits_strictly_on_training_without_future_leakage():
    time_steps, y, x = _make_synthetic_dataset()
    masks = build_labeled_temporal_masks(time_steps, y)

    x_corrupted = x.copy()
    x_corrupted[masks.val_time | masks.test_time] = 1e9

    clean_out = preprocess_features(
        x, masks, feature_set="local_plus_aggregated", fit_on="train_time"
    )
    corrupted_out = preprocess_features(
        x_corrupted, masks, feature_set="local_plus_aggregated", fit_on="train_time"
    )

    assert np.array_equal(clean_out.scaler.mean_, corrupted_out.scaler.mean_)
    assert np.array_equal(clean_out.scaler.scale_, corrupted_out.scaler.scale_)
    assert np.array_equal(clean_out.x[masks.train_time], corrupted_out.x[masks.train_time])
    assert clean_out.scaler.n_samples_seen_ == int(np.sum(masks.train_time))

    train_mean = np.mean(clean_out.x[masks.train_time], axis=0)
    train_std = np.std(clean_out.x[masks.train_time], axis=0)
    assert np.allclose(train_mean, 0.0, atol=1e-5)
    assert np.allclose(train_std, 1.0, atol=1e-5)


def test_scaler_fit_on_train_labeled_excludes_unknown_nodes():
    time_steps, y, x = _make_synthetic_dataset()
    masks = build_labeled_temporal_masks(time_steps, y)

    x_altered = x.copy()
    unknown_train = masks.train_time & ~masks.train_labeled
    x_altered[unknown_train] = -1e7

    clean_out = preprocess_features(x, masks, feature_set="local_only", fit_on="train_labeled")
    altered_out = preprocess_features(
        x_altered, masks, feature_set="local_only", fit_on="train_labeled"
    )

    assert clean_out.scaler.n_samples_seen_ == int(np.sum(masks.train_labeled))
    assert np.array_equal(clean_out.scaler.mean_, altered_out.scaler.mean_)
    assert np.array_equal(clean_out.x[masks.train_labeled], altered_out.x[masks.train_labeled])


def test_constant_training_feature_does_not_produce_nan():
    time_steps, y, x = _make_synthetic_dataset()
    masks = build_labeled_temporal_masks(time_steps, y)

    x[:, 5] = 42.0
    x[masks.test_time, 5] = 100.0

    result = preprocess_features(x, masks, feature_set="local_only", fit_on="train_time")
    assert result.scaler.scale_ is not None
    assert result.scaler.scale_[5] == 1.0
    assert np.isfinite(result.x).all()
    assert np.allclose(result.x[masks.train_time, 5], 0.0)
    assert np.allclose(result.x[masks.test_time, 5], 58.0)


def test_prepare_feature_sets_matches_first_93_columns():
    time_steps, y, x = _make_synthetic_dataset()
    masks = build_labeled_temporal_masks(time_steps, y)

    bundles = prepare_feature_sets(x, masks)
    assert set(bundles.keys()) == {"local_only", "local_plus_aggregated"}
    assert bundles["local_only"].x.shape == (len(x), NUM_LOCAL_FEATURES)
    assert bundles["local_plus_aggregated"].x.shape == (len(x), NUM_TOTAL_FEATURES)
    assert np.allclose(
        bundles["local_only"].x,
        bundles["local_plus_aggregated"].x[:, :NUM_LOCAL_FEATURES],
    )


def test_scaler_rejects_unfitted_transform_and_empty_mask():
    scaler = TrainOnlyStandardScaler()
    x = np.ones((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        scaler.transform(x)
    with pytest.raises(ValueError):
        scaler.fit(x, np.array([False, False, False, False]))


def test_edge_filtering_and_cross_timestep_detection():
    time_steps = np.array([1, 29, 30, 35], dtype=np.int64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    masks = build_labeled_temporal_masks(time_steps, y)

    src = np.array([0, 1, 2, 0], dtype=np.int64)
    dst = np.array([1, 2, 3, 0], dtype=np.int64)

    assert count_cross_timestep_edges(src, dst, time_steps) == 3

    train_src, train_dst = filter_edges_by_mask(src, dst, masks.train_time)
    assert train_src.tolist() == [0, 0]
    assert train_dst.tolist() == [1, 0]

    with pytest.raises(ValueError):
        filter_edges_by_mask(np.array([0, 99]), np.array([0, 1]), masks.train_time)


def test_summarize_split_counts_and_empty_labeled_edge_case():
    time_steps, y, _ = _make_synthetic_dataset()
    masks = build_labeled_temporal_masks(time_steps, y)
    summary = summarize_split(y, masks)

    assert summary["train"] == {
        "total": 4,
        "labeled": 3,
        "unknown": 1,
        "licit": 2,
        "illicit": 1,
        "illicit_rate": pytest.approx(1 / 3),
    }
    assert summary["val"]["labeled"] == 2
    assert summary["test"]["unknown"] == 1

    y_all_unknown = np.full(len(time_steps), UNKNOWN_LABEL, dtype=np.int64)
    masks_unknown = build_labeled_temporal_masks(time_steps, y_all_unknown)
    summary_unknown = summarize_split(y_all_unknown, masks_unknown)
    assert summary_unknown["train"]["illicit_rate"] == 0.0
