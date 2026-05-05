# Myco Fungi Multi-Research Workflow

This repository now follows a multi-experiment autoresearch-style layout.
Core code remains under `src/`, with per-experiment checks colocated under each experiment folder.
The retrieval system in `repos/mycoai_retrieval_backend/` and
`repos/mycoai_retrieval_frontend/` consumes validated outputs from
`src/experiments/retrieval/` and `src/experiments/kmeans_segmentation/`.

This repository is expected to live inside the parent monorepo at `/home/dat/dev/mycoai/`.
Shared runtime paths now live outside this submodule:

- `../Dataset/`
- `../results/`
- `../weights/`
- `../species_weights.json`

`src/config.py` resolves those parent-level paths automatically when this repository is used as the `repos/fungal-cv-qdrant/` submodule.

## High-Level Workflow

1. Put raw data in `../Dataset/original/`
2. Run prepare bootstrap
3. Run one or more experiment programs
4. Run immutable checks from each experiment package
5. Generate analysis visualizations
6. Produce per-experiment LaTeX reports

## Structure

- src/prepare: initialization pipeline and data/qdrant checks
- src/experiments: experiment implementations (preprocessing, feature extraction, finetune_dl, cross_validation, etc.)
- src/utils: reusable helper modules and unified uploader
- src/experiments/*/check.py: concise immutable targets colocated with each experiment
- src/analysis: visualization and analysis scripts
- report: archived markdown reports; new experiments should generate LaTeX reports

## Remote Workspace Bootstrap

When this repository is used inside the MycoAI monorepo, shared remote workspace
bootstrap and dataset sync commands live at the monorepo root under `tools/`.
Run these from `/home/dat/dev/mycoai/`.

The remote workflow assumes you record the Vast.ai `instance_id`, connect with
VSCode Remote-SSH, and keep Google Drive credentials outside the repo via
`RCLONE_CONFIG` or the default `~/.config/rclone/rclone.conf`.

### Prepare and validate a remote workspace

```bash
bash tools/workspace_bootstrap.sh prepare
bash tools/workspace_bootstrap.sh smoke-check
```

`mise install` now installs `rclone` alongside the other shared tools, so the
same prepared workspace can run `tools/dataset_sync.py` without extra package
management.

Use `bash tools/workspace_bootstrap.sh recover --instance-id <id>` after a
restart or replacement if you need to revalidate the workspace and refresh your
local SSH config.

### Preview and run dataset sync

```bash
uv run python tools/dataset_sync.py plan --direction import --remote mydrive:mycoai-dataset --scope original/sample
uv run python tools/dataset_sync.py import --remote mydrive:mycoai-dataset --scope original/sample
uv run python tools/dataset_sync.py export --remote mydrive:mycoai-dataset --scope segmented_image/new-batch
```

The sync CLI uses non-destructive `rclone copy` operations, expects credentials
to live outside the repo, and writes summaries under `results/dataset_sync/`
when run from the monorepo root. Run the `plan` command first to verify remote
access and scope before starting an `import` or `export`.

## Canonical Commands

Run these commands from the monorepo root with `uv --directory fungal-cv-qdrant ...`.

### 1) Full Preparation (Dataset/original -> Qdrant)

```bash
uv --directory fungal-cv-qdrant run python -m src.prepare.init --collection myco_fungi_features_full
```

### 2) Generate Mapping Only

```bash
uv --directory fungal-cv-qdrant run python -m src.utils.generate_strain_mapping
```

### 3) Extract Features Only

```bash
uv --directory fungal-cv-qdrant run python -m src.experiments.feature_extraction.generate_features
```

### 4) Unified Upload (single JSON)

```bash
uv --directory fungal-cv-qdrant run python -m src.utils.upload_qdrant \
  --features-json ../Dataset/segmented_features.json \
  --metadata-json ../Dataset/segmented_image_metadata.json \
  --collection myco_fungi_features_full
```

### 5) Cross Validation

```bash
uv --directory fungal-cv-qdrant run python -m src.experiments.cross_validation.run --collection myco_fungi_features_full_finetuned
uv --directory fungal-cv-qdrant run python -m src.experiments.cross_validation.visualize
```

### 6) Finetune DL

```bash
uv --directory fungal-cv-qdrant run python -m src.experiments.finetune_dl.train_models
```

### 7) Lint / Type Check

```bash
uv --directory fungal-cv-qdrant run black src && uv --directory fungal-cv-qdrant run isort src && uv --directory fungal-cv-qdrant run flake8 src && uv --directory fungal-cv-qdrant run mypy src
```

## Colab

Use src/colab_config.py to print a one-cell setup snippet:

```bash
uv --directory fungal-cv-qdrant run python -m src.colab_config
```

Colab assets are under src/experiments/finetune_dl/colab/.

## Notes

- src/main.py was removed by design.
- Qdrant named vectors must still match extractor names.
- New experiments should include program.md and a colocated check.py target.
