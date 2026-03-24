# Project Context

Myco fungi species classification from colony images using multiple feature extractors (HOG, Gabor, color histograms, ResNet50, MobileNetV2, EfficientNetB1 — ImageNet and finetuned) with Qdrant vector database for KNN retrieval across 5 *Penicillium* species × 7 growth media.

See [`CLAUDE.md`](../CLAUDE.md) for the full architecture, pipeline commands, and conventions.

## Stack

- Python via `uv` (`uv run python ...`, `uv sync`, `uv pip install -r requirements.txt`)
- Qdrant vector DB via Docker (`docker compose up -d`)
- PyTorch for deep learning; scikit-learn for metrics
- Nix shell optional: `nix-shell -r "zsh"`

## Key Conventions

- **Entry point**: `src/main.py` — all pipeline stages are CLI subcommands
- **Config**: `src/config.py` — single source for paths, `QDRANT_URL`, collection names, image size (256×256)
- **Qdrant vector keys** (`using=...`) must match `extractor.name` exactly across feature generation, upload, and query
- **Metadata schema**: `{id, data: {strain, environment, angle, specy}}`; segmented records add `parent_id, segment_index, bbox`
- **Aggregation strategies**: `weighted`/`score`/`avg` = score-weighted; `uni` = uniform-count
- Use Pydantic models for structured data contracts
- When adding dependencies, update both `pyproject.toml` and `requirements.txt`

## Lint / Format

```bash
uv run black src && uv run isort src && uv run flake8 src && uv run mypy src
```
