# Contributing to CryptoGuard++

CryptoGuard++ is developed in small, verifiable milestones. The objective is **correct and reproducible research before complexity**.

## Workflow

1. Select or open a GitHub issue and assign a milestone.
2. Branch from `main`: `feat/m1-data-loader`, `exp/m4-hybrid-ablations`, `fix/temporal-leakage`, or `docs/gcn-explanation`.
3. Scope each PR to one task. Write tests for data processing and bug fixes.
4. Run `python -m pytest -q` and `python -m ruff check .` where applicable.
5. Update README/docs when behavior or protocol changes.
6. Include exact dataset version, split, random seed, model config, environment, metrics, and limitations for an experiment PR.
7. Use a PR for review, then merge. Do not modify `main` directly for unreviewed experimental work.

## Research integrity

- No numbers in README/reports until measured; no invented benchmark improvements.
- Keep train/validation/test isolated. Document graph availability and label masks.
- Preprocessing scalers are fitted on training only.
- Unknown labels are not assumed licit.
- Label an experiment **transductive** if it uses test features/graph topology during representation learning, even without test labels.
- For temporal claims, avoid any information that would not have existed at prediction time.
- Save configurations and commit references for each result.

## Security and data

- Do not commit `data/raw/`, secrets, tokens, `.env`, large model weights, or Kaggle credentials.
- Use `.env.example` to document required local variables without adding values.
- Dataset copyrights and redistribution restrictions must be respected.

## Definition of done

Code + tests + a short explanation + no silently changed evaluation assumptions. If a task is exploratory, mark the outputs as experimental and do not claim them as production-ready.
