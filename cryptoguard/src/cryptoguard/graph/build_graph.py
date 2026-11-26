
import numpy as np
import torch

from torch_geometric.data import Data

from cryptoguard.data.elliptic_loader import EllipticData


def build_graph(dataset: EllipticData) -> Data:

    # 1. Convert node features
    x = torch.from_numpy(dataset.x).float()

    # 2. Convert node labels
    y = torch.from_numpy(dataset.y).long()

    # 3. Construct directed edge index
    edge_index = torch.from_numpy(
        np.stack([dataset.src, dataset.dst])
    ).long()

    # 4. Convert temporal information
    time_step = torch.from_numpy(
        dataset.time_step
    ).long()

    # 5. Preserve original transaction IDs
    tx_id = torch.from_numpy(
        dataset.tx_ids
    ).long()

    # 6. Create PyTorch Geometric graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=time_step,
        tx_id=tx_id,
        num_nodes=x.size(0)
    )

    # 7. Validate graph structure
    graph.validate(raise_on_error=True)

    return graph
