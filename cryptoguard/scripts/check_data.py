
from pathlib import Path
import pandas as pd

# 1. Dataset paths

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "elliptic"

# 2. Load CSV files

features = pd.read_csv(
    DATA_DIR / "elliptic_txs_features.csv",
    header=None
)

edges = pd.read_csv(
    DATA_DIR / "elliptic_txs_edgelist.csv"
)

classes = pd.read_csv(
    DATA_DIR / "elliptic_txs_classes.csv"
)

# 3. Check dimensions

print("=== DATASET SHAPES ===")
print("Features:", features.shape)
print("Edges:", edges.shape)
print("Classes:", classes.shape)

# 4. Check labels

print("\n=== CLASS DISTRIBUTION ===")
print(classes["class"].value_counts())

# 5. Check temporal information

time_steps = features.iloc[:, 1]

print("\n=== TEMPORAL INFORMATION ===")
print("Number of time steps:", time_steps.nunique())
print("First time step:", time_steps.min())
print("Last time step:", time_steps.max())

# 6. Validate dataset integrity

assert features.shape == (203769, 167)
assert edges.shape == (234355, 2)
assert classes.shape == (203769, 2)

assert features.iloc[:, 0].is_unique
assert classes["txId"].is_unique

assert time_steps.nunique() == 49
assert time_steps.min() == 1
assert time_steps.max() == 49

# Every transaction must have a matching class entry
feature_ids = set(features.iloc[:, 0])
class_ids = set(classes["txId"])

assert feature_ids == class_ids

# Every edge must connect valid transaction IDs
edge_ids = set(edges.iloc[:, 0]) | set(edges.iloc[:, 1])

assert edge_ids.issubset(feature_ids)

print("\nALL VALIDATION CHECKS PASSED!")
