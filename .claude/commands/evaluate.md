Run the standard evaluation with the finetuned collection using EfficientNetB1 finetuned extractor.

Make sure Qdrant is running first:
```bash
docker compose up -d
```

Then run:
```bash
uv run python src/main.py evaluate \
  --extractor efficientnetb1_finetuned \
  --k 7 \
  --strategy weighted \
  --environment all \
  --collection myco_fungi_features_full_finetuned
```

To evaluate all extractor/strategy combinations:
```bash
uv run python src/main.py evaluate-all \
  --k 7 \
  --collection myco_fungi_features_full_finetuned
```

Results are saved to `results/run_<timestamp>_k<K>/`.
