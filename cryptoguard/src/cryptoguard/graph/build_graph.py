"""PyTorch Geometric directed graph builder for the Elliptic dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from cryptoguard.data.elliptic_loader import (
    EllipticData,
    validate_edge_indices,
    validate_temporal_edges,
    validate_time_steps,
    validate_transaction_ids,
)
from cryptoguard.data.validation import validate_strict_int64


def build_graph(dataset: EllipticData) -> Data:
    """Construct and validate a directed PyTorch Geometric graph from ``EllipticData``."""
    # 1. Validate node features into an owned, writable float32 array
    x_np = np.array(dataset.x, dtype=np.float32, copy=True, order="C")
    if x_np.ndim != 2:
        raise ValueError(f"Expected 2D node feature matrix, got shape {x_np.shape}.")
    if not np.isfinite(x_np).all():
        raise ValueError("Non-finite feature values detected")

    num_nodes = int(x_np.shape[0])

    # 2. Strictly validate integer node attributes before tensor conversion
    validated_time_step = validate_time_steps(dataset.time_step)
    validated_y = validate_strict_int64(dataset.y, name="y", allow_integral_floats=False)
    validated_tx_ids = validate_transaction_ids(dataset.tx_ids, name="tx_id")

    for name, arr in (
        ("y", validated_y),
        ("time_step", validated_time_step),
        ("tx_id", validated_tx_ids),
    ):
        if arr.ndim != 1 or len(arr) != num_nodes:
            raise ValueError(
                f"Node attribute '{name}' must have shape ({num_nodes},), "
                f"got {arr.shape}."
            )

    if len(validated_tx_ids) > 1 and len(np.unique(validated_tx_ids)) != len(validated_tx_ids):
        unique_ids, counts = np.unique(validated_tx_ids, return_counts=True)
        dup_mask = counts > 1
        dup_values = unique_ids[dup_mask][:5].tolist()
        raise ValueError(
            f"Duplicate tx_id values are not permitted: found {int(np.sum(dup_mask))} "
            f"duplicate ID(s); examples: {dup_values}."
        )

    # 3. Strictly validate directed edge indices before casting to int64 tensors
    src_np, dst_np = validate_edge_indices(dataset.src, dataset.dst, num_nodes=num_nodes)

    # 4. Validate temporal edge integrity (reject cross-time-step edges)
    validate_temporal_edges(src_np, dst_np, validated_time_step, tx_ids=validated_tx_ids)

    # 5. Construct owned, writable tensors (no shared/read-only NumPy memory)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(validated_y)
    time_step = torch.from_numpy(validated_time_step)
    tx_id = torch.from_numpy(validated_tx_ids)
    edge_index_np = np.ascontiguousarray(np.stack([src_np, dst_np], axis=0), dtype=np.int64)
    edge_index = torch.from_numpy(edge_index_np)

    # 6. Create and validate PyTorch Geometric directed graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=time_step,
        tx_id=tx_id,
        num_nodes=num_nodes,
    )
    graph.validate(raise_on_error=True)

    return graph
