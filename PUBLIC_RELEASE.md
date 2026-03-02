# Public Release Scope

This document defines what is safe to publish to GitHub for this project.

## Include in public repo

- `src/config.py`
- `src/openai_client.py`
- `src/core/*.py`
- `src/cli/*.py`
- `src/cli/suites/*.json`
- `src/app/*.py`
- `src/app/demo/*`
- `tests/*.py`
- `README.md`
- `Makefile`
- `.github/workflows/ci.yml`
- `requirements.txt`
- `data/sample_products.csv`
- `data/sample_queries_train.csv`
- `data/sample_queries_test.csv`
- `data/sample_submission.csv`
- `data/README.md`
- `.gitignore`

## Exclude from public repo

- `products.csv`
- `queries_train.csv`
- `queries_test.csv`
- `cache/`
- `outputs/`
- `gitleaks-*.json` (local scan reports)
- `.env`, `.env.local`
- `__pycache__/`
- `.DS_Store`

## Notes

- Real datasets stay local/private.
- Generated outputs and embedding caches stay local/private.
- Public sample CSVs are synthetic and intended only for demo/testing.
