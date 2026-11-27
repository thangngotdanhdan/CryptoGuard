import csv
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cryptoguard.data.elliptic_loader import load_elliptic, validate_temporal_edges


TX_IDS = [2**53 + 3, 70, 9, 2**53 + 4]
STEPS = [30, 1, 1, 49]
FEATURES = np.arange(4 * 165, dtype=np.float32).reshape(4, 165) + 0.25
LABELS = [-1, 1, 0, 1]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


@pytest.fixture
def elliptic_csv(tmp_path):
    write_csv(
        tmp_path / "elliptic_txs_features.csv",
        ([tx_id, step, *row] for tx_id, step, row in zip(TX_IDS, STEPS, FEATURES)),
    )
    write_csv(
        tmp_path / "elliptic_txs_classes.csv",
        [["txId", "class"], [9, "2"], [TX_IDS[3], "1"], [TX_IDS[0], "unknown"], [70, "1"]],
    )
    write_csv(tmp_path / "elliptic_txs_edgelist.csv", [["txId1", "txId2"], [9, 70]])
    return tmp_path


def test_loader_aligns_features_labels_times_and_directed_edges(elliptic_csv):
    dataset = load_elliptic(elliptic_csv)

    assert dataset.tx_to_idx == {tx_id: index for index, tx_id in enumerate(TX_IDS)}
    np.testing.assert_array_equal(dataset.tx_ids, TX_IDS)
    np.testing.assert_array_equal(dataset.x, FEATURES)
    np.testing.assert_array_equal(dataset.y, LABELS)
    np.testing.assert_array_equal(dataset.time_step, STEPS)
    np.testing.assert_array_equal(dataset.src, [2])
    np.testing.assert_array_equal(dataset.dst, [1])
    assert dataset.x.dtype == np.float32
    for array in (dataset.y, dataset.time_step, dataset.tx_ids, dataset.src, dataset.dst):
        assert array.dtype == np.int64
    for array in (dataset.x, dataset.y, dataset.time_step, dataset.tx_ids, dataset.src, dataset.dst):
        assert array.flags.writeable
        assert array.flags.owndata


def test_large_transaction_ids_round_trip_through_edge_mapping(elliptic_csv):
    # Consecutive integers beyond float64's exact range must stay distinct at both endpoints.
    steps = [30, 1, 1, 30]
    write_csv(
        elliptic_csv / "elliptic_txs_features.csv",
        ([tx_id, step, *row] for tx_id, step, row in zip(TX_IDS, steps, FEATURES)),
    )
    write_csv(
        elliptic_csv / "elliptic_txs_edgelist.csv",
        [["txId1", "txId2"], [TX_IDS[3], TX_IDS[0]], [9, 70]],
    )
    dataset = load_elliptic(elliptic_csv)
    np.testing.assert_array_equal(dataset.tx_ids[dataset.src], [TX_IDS[3], 9])
    np.testing.assert_array_equal(dataset.tx_ids[dataset.dst], [TX_IDS[0], 70])


def test_loader_rejects_ids_that_collapse_to_the_same_integer(elliptic_csv):
    # Both keys are distinct when merged but collide if truncated for tx_to_idx.
    write_csv(
        elliptic_csv / "elliptic_txs_features.csv",
        [[17.1, 1, *FEATURES[0]], [17.9, 1, *FEATURES[1]]],
    )
    write_csv(
        elliptic_csv / "elliptic_txs_classes.csv",
        [["txId", "class"], [17.9, "2"], [17.1, "1"]],
    )
    write_csv(elliptic_csv / "elliptic_txs_edgelist.csv", [["txId1", "txId2"]])
    with pytest.raises(ValueError):
        load_elliptic(elliptic_csv)


def test_loader_rejects_duplicate_feature_transaction_id(elliptic_csv):
    path = elliptic_csv / "elliptic_txs_features.csv"
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([TX_IDS[0], STEPS[0], *FEATURES[1]])

    with pytest.raises(ValueError):
        load_elliptic(elliptic_csv)


@pytest.mark.parametrize(
    ("table", "row", "column"),
    [("features", 1, 0), ("classes", 4, 0), ("edgelist", 1, 0), ("edgelist", 1, 1)],
)
def test_loader_rejects_fractional_id_before_it_rounds_to_a_valid_id(
    elliptic_csv, table, row, column
):
    path = elliptic_csv / f"elliptic_txs_{table}.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    # Parsing this token as float64 would give back the original valid endpoint or join key.
    rows[row][column] += ".0000000000000000001"
    write_csv(path, rows)

    with pytest.raises(ValueError):
        load_elliptic(elliptic_csv)


@pytest.mark.parametrize(
    ("filename", "rows", "error", "message"),
    [
        ("features", [[70, 1, 0]], ValueError, "Invalid feature dimensions"),
        ("classes", [["txId", "class"]], ValueError, "Missing transaction labels"),
        (
            "classes",
            [["txId", "class"], [70, "3"], [9, "2"], [TX_IDS[0], "unknown"], [TX_IDS[3], "1"]],
            ValueError,
            "Invalid class label",
        ),
        (
            "classes",
            [["txId", "class"], [70, "1"], [70, "2"]],
            pd.errors.MergeError,
            "not a one-to-one merge",
        ),
        ("edgelist", [["txId1", "txId2"], [404, 70]], ValueError, "Invalid edge endpoints"),
        ("edgelist", [["txId1", "txId2"], [70, 404]], ValueError, "Invalid edge endpoints"),
    ],
)
def test_loader_rejects_invalid_csv(elliptic_csv, filename, rows, error, message):
    write_csv(elliptic_csv / f"elliptic_txs_{filename}.csv", rows)
    with pytest.raises(error, match=message):
        load_elliptic(elliptic_csv)


def test_loader_rejects_nonfinite_features(elliptic_csv):
    path = elliptic_csv / "elliptic_txs_features.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    rows[0][2] = "inf"
    write_csv(path, rows)

    with pytest.raises(ValueError, match="Non-finite feature values"):
        load_elliptic(elliptic_csv)


@pytest.mark.parametrize("empty_edges", [False, True])
def test_graph_preserves_node_rows_and_edge_direction(elliptic_csv, empty_edges):
    import torch

    from cryptoguard.graph.build_graph import build_graph

    if empty_edges:
        write_csv(elliptic_csv / "elliptic_txs_edgelist.csv", [["txId1", "txId2"]])
    graph = build_graph(load_elliptic(elliptic_csv))

    assert graph.num_nodes == len(TX_IDS)
    torch.testing.assert_close(graph.x, torch.tensor(FEATURES, dtype=torch.float32))
    torch.testing.assert_close(graph.y, torch.tensor(LABELS, dtype=torch.long))
    torch.testing.assert_close(graph.time_step, torch.tensor(STEPS, dtype=torch.long))
    torch.testing.assert_close(graph.tx_id, torch.tensor(TX_IDS, dtype=torch.long))
    expected = torch.empty((2, 0), dtype=torch.long) if empty_edges else torch.tensor([[2], [1]])
    torch.testing.assert_close(graph.edge_index, expected)


def test_graph_rejects_out_of_bounds_edge(elliptic_csv):
    from cryptoguard.graph.build_graph import build_graph

    dataset = load_elliptic(elliptic_csv)
    dataset = replace(dataset, dst=np.array([len(TX_IDS)], dtype=np.int64))
    with pytest.raises(ValueError, match="edge_index"):
        build_graph(dataset)


@pytest.mark.parametrize(("source", "target"), [(3, 1), (1, 3)])
def test_loader_rejects_cross_timestep_edges(elliptic_csv, source, target):
    # Include a valid edge first so that diagnostics must identify the actual offender.
    write_csv(
        elliptic_csv / "elliptic_txs_edgelist.csv",
        [["txId1", "txId2"], [9, 70], [TX_IDS[source], TX_IDS[target]]],
    )
    with pytest.raises(ValueError, match=r"(?i)cross[- ]time[- ]step") as exc:
        load_elliptic(elliptic_csv)
    message = str(exc.value)
    for node in (source, target):
        assert f"txId {TX_IDS[node]}" in message
        assert f"time_step={STEPS[node]}" in message


def test_temporal_edge_diagnostic_without_transaction_ids():
    with pytest.raises(ValueError, match="Cross-time-step") as exc:
        validate_temporal_edges(np.array([0, 2, 1]), np.array([1, 1, 2]), np.array([1, 1, 2]))
    message = str(exc.value)
    assert "2 invalid directed edge" in message
    assert "node 2 (time_step=2) -> node 1 (time_step=1)" in message


@pytest.mark.parametrize(("src", "dst"), [([[0]], [1]), ([0], [[1]]), ([0, 1], [1])])
def test_temporal_edge_validator_rejects_incompatible_shapes(src, dst):
    with pytest.raises(ValueError, match="edge arrays"):
        validate_temporal_edges(np.array(src), np.array(dst), np.array([1, 1]))
