# =================================================
# CryptoGuard++ Dataset
# =================================================

# Download the Elliptic Bitcoin Dataset from:
# https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

# Extract the following files into data/raw/elliptic/:
# - elliptic_txs_features.csv
# - elliptic_txs_edgelist.csv
# - elliptic_txs_classes.csv

# Raw and processed datasets
data/raw/**
data/processed/**

# Keep directory placeholders and documentation
!data/raw/.gitkeep
!data/raw/README.md
!data/processed/.gitkeep

# Python virtual environments
.venv/
venv/

# Python cache
__pycache__/
*.pyc

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Environment variables and credentials
.env
kaggle.json