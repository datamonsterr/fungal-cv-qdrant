# Project Context

Myco fungi species classification from colony images using multiple feature extractors and Qdrant vector search.

## Architecture Direction

The repository is organized for multi-experiment autoresearch workflows:

- src/prepare: bootstrap from raw dataset to ready artifacts and upload
- src/experiments: experiment implementations by use case
- src/utils: shared utilities (including unified upload_qdrant)
- src/experiments/*/check.py: immutable experiment targets colocated with each experiment
- src/analysis: custom visualizations

## Stack

- Python via uv
- Qdrant vector DB
- PyTorch and scikit-learn

## Key Conventions

- No src/main.py entrypoint
- Use module entrypoints with python -m
- Qdrant named-vector keys must match extractor names across generation/upload/query
- Keep config centralized in src/config.py
- Add dependencies to pyproject.toml and requirements.txt
