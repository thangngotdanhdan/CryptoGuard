
from pathlib import Path
import torch

from cryptoguard.data.elliptic_loader import load_elliptic
from cryptoguard.graph.build_graph import build_graph


root = Path(__file__).resolve().parents[1]

data_dir = root / "data" / "raw" / "elliptic"

# 1. Load dataset
dataset = load_elliptic(data_dir)

# 2. Construct graph
graph = build_graph(dataset)

# 3. Display graph information
print("=== GRAPH INFORMATION ===")
print(graph)

print("\n=== GRAPH DIMENSIONS ===")
print("Nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)
print("Features:", graph.num_node_features)

print("\n=== TENSOR SHAPES ===")
print("x:", graph.x.shape)
print("y:", graph.y.shape)
print("edge_index:", graph.edge_index.shape)
print("time_step:", graph.time_step.shape)

# 4. Verify dimensions
assert graph.num_nodes == 203769
assert graph.num_edges == 234355
assert graph.num_node_features == 165

assert graph.x.shape == (203769, 165)
assert graph.y.shape == (203769,)
assert graph.edge_index.shape == (2, 234355)

# 5. Verify data types
assert graph.x.dtype == torch.float32
assert graph.y.dtype == torch.long
assert graph.edge_index.dtype == torch.long

# 6. Verify labels
assert (graph.y == -1).sum().item() == 157205
assert (graph.y == 0).sum().item() == 42019
assert (graph.y == 1).sum().item() == 4545

# 7. Verify temporal consistency
src, dst = graph.edge_index

assert torch.equal(
    graph.time_step[src],
    graph.time_step[dst]
)

print("\nALL GRAPH VALIDATION CHECKS PASSED!")
