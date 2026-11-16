# How to collaborate on CryptoGuard++ with ChatGPT

## Ground rules

- One GitHub issue per specific task; use a new branch for each code change.
- The project owner controls GitHub permissions, merging, secrets, and deployment.
- ChatGPT can help explain concepts, write/review code, propose tests, debug errors, and draft PR descriptions. Access to GitHub files/actions depends on a connected integration or the files you provide.
- Keep **a single source of truth** in GitHub: the code and the dated experiment protocol. Update it after each milestone.

## Recommended issue template for a ChatGPT request

```text
Project: CryptoGuard++
Milestone: M1 (Elliptic Data Pipeline)
Task: Add a reliable CSV loader with ID validation
Branch: feat/m1-data-loader
Relevant files: src/cryptoguard/data/, tests/
Current behavior: [what happens now]
Expected behavior: [explicit acceptance criteria]
Input sample: [small synthetic example, no private data]
Actual error / logs: [full traceback if any]
Constraints: Python version, Colab/OS, no leakage
What I need: Code diff + unit tests + explanation
```

## After receiving an answer

1. Copy the proposed files/diff into your feature branch (or review changes through your connected GitHub tool if one is available).
2. Run all relevant tests and a minimal reproducible experiment.
3. Return errors and changed files for another iteration.
4. Open a PR and attach screenshots/metrics when appropriate.
5. Do not paste credentials or private keys into chat or GitHub.

## Suggested first GitHub issues

- `M0: Verify notebook 00_week1_gcn_foundations.ipynb in Colab`
- `M1: Implement Elliptic CSV loader + ID mapping + schema tests`
- `M1: Produce chronological split and label-mask validation`
- `M1: Create label-count-by-time EDA report`
- `M2: Train and evaluate RF baseline with fixed protocol`
- `M2: Train and evaluate XGBoost baseline`
- `M3: Build GCN node classifier in PyG`

## Branch naming

- `feat/mX-description` for features
- `exp/mX-description` for experiments
- `fix/description` for bugs
- `docs/description` for documents

## ChatGPT context recovery

When starting a new chat, share **the GitHub repo URL, current milestone, relevant files, last validated results, and the issue you want to work on**. Do not assume every past local file or uncommitted decision is accessible in a new session.
