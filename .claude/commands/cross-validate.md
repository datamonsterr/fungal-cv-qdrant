Run or resume cross-validation (100 runs) and visualize results.

Make sure Qdrant is running first:
```bash
docker compose up -d
```

**Run cross-validation** (safe to interrupt and resume — results append to CSV):
```bash
uv run python src/main.py cross-validate \
  --collection myco_fungi_features_full_finetuned
```

**Visualize results after completion:**
```bash
uv run python src/main.py cross-validate-visualize
```

Results are written to `results/cross_validation/`.
