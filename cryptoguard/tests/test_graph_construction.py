from dataclasses import replace

import numpy as np
import pytest

from cryptoguard.data.elliptic_loader import EllipticData


@pytest.fixture
def transactions():
    # Node 3 is deliberately isolated. A node count inferred from edges would lose it.
    ids = np.array([87, 12, 4, 2**53 + 7], dtype=np.int64)
    return EllipticData(
        x=np.arange(12, dtype=np.float32).reshape(4, 3),
        y=np.array([1, 0, -1, 1], dtype=np.int64),
        src=np.array([2, 0], dtype=np.int64),
        dst=np.array([1, 2], dtype=np.int64),
        time_step=np.array([3, 3, 3, 49], dtype=np.int64),
        tx_ids=ids,
        tx_to_idx={int(tx_id): i for i, tx_id in enumerate(ids)},
    )


def test_graph_keeps_directed_edge_order_and_isolated_node(transactions):
    from cryptoguard.graph.build_graph import build_graph

    graph = build_graph(transactions)
    assert graph.num_nodes == 4
    assert graph.tx_id.tolist() == [87, 12, 4, 2**53 + 7]
    assert graph.edge_index.tolist() == [[2, 0], [1, 2]]
    assert graph.tx_id[graph.edge_index].tolist() == [[4, 87], [12, 4]]


def test_zero_nodes_and_zero_edges(transactions):
    from cryptoguard.graph.build_graph import build_graph

    empty = EllipticData(
        x=transactions.x[:0],
        y=transactions.y[:0],
        src=transactions.src[:0],
        dst=transactions.dst[:0],
        time_step=transactions.time_step[:0],
        tx_ids=transactions.tx_ids[:0],
        tx_to_idx={},
    )
    graph = build_graph(empty)
    assert graph.num_nodes == graph.num_edges == 0
    assert graph.x.shape == (0, 3)
    assert graph.edge_index.shape == (2, 0)
    assert graph.tx_id.shape == graph.y.shape == graph.time_step.shape == (0,)


@pytest.mark.parametrize("read_only", [False, True], ids=["writable", "read-only"])
def test_graph_storage_is_independent_of_numpy_inputs(transactions, read_only):
    from cryptoguard.graph.build_graph import build_graph

    sources = (
        transactions.x, transactions.y, transactions.time_step, transactions.tx_ids,
        transactions.src, transactions.dst,
    )
    if read_only:
        for source in sources:
            source.setflags(write=False)

    graph = build_graph(transactions)
    tensors = (graph.x, graph.y, graph.time_step, graph.tx_id, *graph.edge_index)
    for source, tensor in zip(sources, tensors, strict=True):
        original = source.copy()
        assert source.flags.writeable == (not read_only)
        np.testing.assert_array_equal(tensor.numpy(), original)
        assert not np.shares_memory(tensor.numpy(), source)
        tensor.fill_(77)
        np.testing.assert_array_equal(source, original)
        if not read_only:
            source.fill(33)
            assert (tensor == 77).all().item()


@pytest.mark.parametrize(("source", "target"), [(0, 3), (3, 0)])
def test_graph_rejects_cross_time_edges_with_transaction_details(transactions, source, target):
    from cryptoguard.graph.build_graph import build_graph

    # Keep the first edge valid to ensure the builder checks the whole graph.
    invalid = replace(transactions, src=np.array([2, source]), dst=np.array([1, target]))
    with pytest.raises(ValueError, match="(?i)cross-time-step") as caught:
        build_graph(invalid)
    message = str(caught.value)
    for node in (source, target):
        assert f"txId {transactions.tx_ids[node]}" in message
        assert f"time_step={transactions.time_step[node]}" in message


@pytest.mark.parametrize("endpoint", ["src", "dst"])
@pytest.mark.parametrize("index", [-1, 4])
def test_graph_rejects_invalid_node_indices(transactions, endpoint, index):
    from cryptoguard.graph.build_graph import build_graph

    invalid = replace(transactions, **{endpoint: np.array([index, 0])})
    with pytest.raises(ValueError, match="edge_index"):
        build_graph(invalid)


@pytest.mark.parametrize(
    ("endpoint", "indices"),
    [
        # Truncating -0.5 creates a valid but spurious edge from node zero.
        pytest.param("src", np.array([-0.5, 0]), id="fractional"),
        pytest.param("dst", np.array([1.5, 2.0]), id="fractional-destination"),
        pytest.param("src", np.array([True, 0], dtype=object), id="mixed-boolean"),
        pytest.param("dst", np.array([1, np.inf]), id="nonfinite"),
        pytest.param("src", np.array([2**63, 0], dtype=np.uint64), id="uint64-overflow"),
    ],
)
def test_graph_rejects_invalid_edge_values_before_casting(transactions, endpoint, indices):
    from cryptoguard.graph.build_graph import build_graph

    invalid = replace(transactions, **{endpoint: indices})
    with pytest.raises(ValueError, match="edge_index"):
        build_graph(invalid)


@pytest.mark.parametrize("endpoint", ["src", "dst"])
def test_graph_rejects_masked_edge_index(transactions, endpoint):
    from cryptoguard.graph.build_graph import build_graph

    # All backing indices are valid; only the mask makes the second edge invalid.
    indices = np.ma.array(getattr(transactions, endpoint), mask=[False, True])
    with pytest.raises(ValueError, match=r"edge_index.*(?:missing|masked)"):
        build_graph(replace(transactions, **{endpoint: indices}))


def test_graph_rejects_duplicate_transaction_ids(transactions):
    from cryptoguard.graph.build_graph import build_graph

    ids = transactions.tx_ids.copy()
    ids[0] = ids[-1]
    invalid = replace(transactions, tx_ids=ids, tx_to_idx={int(tx): i for i, tx in enumerate(ids)})
    with pytest.raises(ValueError, match="(?i)duplicate.*tx_id"):
        build_graph(invalid)


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([3, 3, 3, 48.5], id="fractional"),
        pytest.param([True] * 4, id="boolean"),
        pytest.param([3, 3, 3, 50], id="out-of-range"),
    ],
)
def test_graph_revalidates_time_steps_on_isolated_nodes(transactions, steps):
    from cryptoguard.graph.build_graph import build_graph

    with pytest.raises(ValueError, match="time_step"):
        build_graph(replace(transactions, time_step=np.array(steps)))
