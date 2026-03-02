# Public Sample Data

This folder contains synthetic sample data for public repository usage.

- `sample_products.csv`: product catalog sample
- `sample_queries_train.csv`: sample training queries with labels
- `sample_queries_test.csv`: sample test queries
- `sample_submission.csv`: sample prediction output format

## Why these files exist

The original project data files are intentionally excluded from version control for privacy and distribution control.

Excluded private files at project root:
- `products.csv`
- `queries_train.csv`
- `queries_test.csv`

The code will automatically use these sample files when private root-level files are missing.

## Using private data locally

Set environment variables to your private file paths:

```bash
export PRODUCTS_CSV=/path/to/products.csv
export QUERIES_TRAIN_CSV=/path/to/queries_train.csv
export QUERIES_TEST_CSV=/path/to/queries_test.csv
```
