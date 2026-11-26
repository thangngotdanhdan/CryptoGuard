from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EllipticData:
    x: np.ndarray
    y: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    time_step: np.ndarray
    tx_ids: np.ndarray
    tx_to_idx: dict

def load_elliptic(data_dir: str | Path) -> EllipticData:

    data_dir = Path(data_dir)

    # 1. Load raw CSV files
    features = pd.read_csv(
        data_dir / "elliptic_txs_features.csv",
        header=None
    )

    classes = pd.read_csv(
        data_dir / "elliptic_txs_classes.csv",
        dtype={"class": str}
    )

    edges = pd.read_csv(
        data_dir / "elliptic_txs_edgelist.csv"
    )

    # 2. Validate feature dimensions
    if features.shape[1] != 167:
        raise ValueError("Invalid feature dimensions")

    # 3. Name feature columns
    feature_cols = [
        f"feature_{i}" for i in range(1, 166)
    ]

    features.columns = [
        "txId",
        "time_step",
        *feature_cols
    ]

    # 4. Merge features and labels
    df = features.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    if df["class"].isna().any():
        raise ValueError("Missing transaction labels")

    # 5. Encode labels
    label_mapping = {
        "1": 1,
        "2": 0,
        "unknown": -1
    }

    df["label"] = df["class"].map(label_mapping)

    if df["label"].isna().any():
        raise ValueError("Invalid class label")

    # 6. Map transaction IDs to node indices
    tx_ids = df["txId"].to_numpy(dtype=np.int64)

    tx_to_idx = {
        int(tx_id): idx
        for idx, tx_id in enumerate(tx_ids)
    }

    # 7. Map edge endpoints
    src = edges["txId1"].map(tx_to_idx)
    dst = edges["txId2"].map(tx_to_idx)

    if src.isna().any() or dst.isna().any():
        raise ValueError("Invalid edge endpoints")

    # 8. Convert to NumPy arrays
    x = df[feature_cols].to_numpy(dtype=np.float32)

    y = df["label"].to_numpy(dtype=np.int64)

    time_step = df["time_step"].to_numpy(
        dtype=np.int64
    )

    src = src.to_numpy(dtype=np.int64)
    dst = dst.to_numpy(dtype=np.int64)

    # 9. Validate numerical features
    if not np.isfinite(x).all():
        raise ValueError("Non-finite feature values detected")

    # 10. Return processed data
    return EllipticData(
        x=x,
        y=y,
        src=src,
        dst=dst,
        time_step=time_step,
        tx_ids=tx_ids,
        tx_to_idx=tx_to_idx
    )
