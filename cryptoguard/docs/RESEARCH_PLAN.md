# Research plan — CryptoGuard++

## Hypothesis

A graph encoder may produce complementary representations to anonymized tabular transaction features, and concatenating its node embeddings with raw features **may** improve XGBoost illicit-class PR-AUC and Recall@K on future time windows.

## Null hypothesis

Under a strict chronological protocol, hybrid graph embeddings do not add measurable predictive value over strong tabular baselines.

## Experimental design

1. Logistic Regression, Random Forest, XGBoost on raw features.
2. GCN, GraphSAGE, GAT as graph-based node classifiers.
3. GraphSAGE→GAT node encoder with learned embeddings fed into XGBoost.
4. Ablations: raw only; embeddings only; raw + embeddings; raw + structural statistics; raw local-only features if feasible.
5. Optional Elliptic++ experiment with transaction and wallet/address node types, after establishing stable original-Elliptic results.

## Main metrics

Illicit-class PR-AUC; F1, precision, recall with validation-chosen threshold; Recall@K and Precision@K; calibration where appropriate; p95 inference latency and resource use.

## Result integrity

No model can be called "better" until evaluated with the same observed-information policy, test population, and split. Negative results are valid research outcomes. Record training compute and model complexity, not only accuracy.

## Paper outline

1. Introduction and motivation
2. Related work and dataset caveats
3. Research question, hypothesis, and evaluation protocol
4. Models and hybrid design
5. Baseline and ablation results
6. Robustness, leakage, explanations, and limitations
7. Engineering prototype and conclusion
