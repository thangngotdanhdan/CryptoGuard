# Evaluation and leakage prevention protocol — v0.1 (proposed)

This protocol is a starting point, not a published Elliptic benchmark standard. **Freeze a chosen protocol before comparing models.** Document every deviation explicitly.

## Task

- **Target:** binary illicit/licit transaction classification on **labeled** nodes.
- **Unknown:** unlabeled. May be used as graph context only if the experimental protocol allows those nodes and their features at the prediction time; never convert unknown into a negative ground-truth label.
- **Split:** illustrative train steps 1–29; validation 30–34; test 35–49.

## Availability is more important than the split mask

A temporal train mask is insufficient if the encoder sees features, edges, derived aggregates, or labels from future times. Before each experiment, write down:

1. When the prediction is made (node arrival, snapshot close, or end of the period).
2. Which nodes, directed edges, and original/aggregate features are observable at that point.
3. Whether graph propagation includes held-out time steps.
4. Whether feature engineering uses labels or summaries of future nodes.
5. Whether evaluation is inductive (future graph excluded at training) or transductive (some test graph may be observed). **Do not compare protocols as if identical.**

Note: the original Elliptic dataset includes aggregated transaction features. Their timing and derivation may affect whether a strict point-in-time interpretation is justified. Treat any unresolved availability issue as a documented limitation, not as established leakage-freedom.

## Safeguards

- Fit normalization, imputation, feature selection, and dimensionality reduction on training only.
- Never tune hyperparameters/thresholds against final test results.
- For a two-stage GNN→XGBoost approach, record how training embeddings were generated and whether they are in-sample. Consider out-of-fold/rolling-origin training embeddings or a dedicated encoder-fit split.
- For hybrid vs baseline comparisons use the same labeled evaluation nodes, time windows, and metrics.
- Fix and log seeds, dependency versions, class weights, sampling method, and model checkpoints.
- Report experiments across multiple seeds, with variability rather than a cherry-picked best run.

## Minimum report

| Experiment | Feature set | Learning protocol | Val PR-AUC | Test PR-AUC | Test illicit F1 | Recall@K | Latency |
|---|---|---|---|---|---|---|---|
| RF | Raw | To be specified | TBD | TBD | TBD | TBD | TBD |
| XGBoost | Raw | To be specified | TBD | TBD | TBD | TBD | TBD |
| GCN | Graph + features | To be specified | TBD | TBD | TBD | TBD | TBD |
| GraphSAGE | Graph + features | To be specified | TBD | TBD | TBD | TBD | TBD |
| GAT | Graph + features | To be specified | TBD | TBD | TBD | TBD | TBD |
| GraphSAGE–GAT → XGBoost | Raw + learned embeddings | To be specified | TBD | TBD | TBD | TBD | TBD |

## Meaning of predictions

Scores are **screening signals**, not proof of illegal activity. Rank cases for human analysis and discuss false-positive costs and calibration limits.
