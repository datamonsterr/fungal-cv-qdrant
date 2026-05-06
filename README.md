# Myco Fungi Multi-Research Workflow

This repository now follows a multi-experiment autoresearch-style layout.
Core code remains under `src/`, with per-experiment checks colocated under each experiment folder.
The retrieval system in `mycoai_retrieval_backend/` and
`mycoai_retrieval_frontend/` consumes validated outputs from
`src/experiments/retrieval/` and `src/experiments/kmeans_segmentation/`.

This repository is expected to live inside the parent monorepo at `/home/dat/dev/mycoai/`.
Shared runtime paths now live outside this submodule:

- `../Dataset/`
- `../results/`
- `../weights/`
- `../species_weights.json`

`src/config.py` resolves those parent-level paths automatically when this repository is used as the `fungal-cv-qdrant/` submodule.

## High-Level Workflow

1. Put raw data in `../Dataset/curated_primary/` and/or `../Dataset/incoming_low_quality/`
2. Run prepare bootstrap (canonical hierarchy + segmentation + features)
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

## Canonical Dataset Layout

After restructure, the dataset uses a canonical hierarchy:

```
Dataset/
├── curated_primary/              # Primary curated source images (equiv. to original/)
├── incoming_low_quality/         # Lower-quality incoming source images (equiv. to new_data/)
└── prepared/                     # Canonical derived hierarchy
    └── {species}/
        └── {strain}/
            └── {environment}/
                └── {image_stem}/
                    ├── source.jpg
                    ├── prepared.jpg
                    ├── item.json
                    ├── segments_kmeans/seg_0.jpg, seg_1.jpg, seg_2.jpg
                    ├── segments_contour/seg_0.jpg, ...
                    ├── bbox_kmeans.jpg
                    ├── bbox_contour.jpg
                    ├── pipeline_kmeans.jpg
                    └── pipeline_contour.jpg
```

Metadata records carry exact canonical paths. Consumers read `segment_path` from
`Dataset/prepared_segments_metadata.json` instead of constructing flat directory paths.

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
uv run python tools/dataset_sync.py plan --direction import --remote mydrive:mycoai-dataset --scope curated_primary/sample
uv run python tools/dataset_sync.py import --remote mydrive:mycoai-dataset --scope curated_primary/sample
uv run python tools/dataset_sync.py export --remote mydrive:mycoai-dataset --scope prepared/segments
```

The sync CLI uses non-destructive `rclone copy` operations, expects credentials
to live outside the repo, and writes summaries under `results/dataset_sync/`
when run from the monorepo root. Run the `plan` command first to verify remote
access and scope before starting an `import` or `export`.

## Canonical Commands

Run these commands from the monorepo root with `uv --directory fungal-cv-qdrant ...`.

### 1) Full Preparation (Source Collections -> Canonical Hierarchy + Segments + Features)

```bash
uv --directory fungal-cv-qdrant run python -m src.prepare.init --collection myco_fungi_features_full
```

Options: `--source-collection curated`, `--source-collection incoming`, `--limit N` for smoke runs.

### 2) Generate Mapping Only

```bash
uv --directory fungal-cv-qdrant run python -m src.utils.generate_strain_mapping
```

### 3) Analyze Source Datasets

```bash
uv --directory fungal-cv-qdrant run python -m src.analysis.dataset_eda --format json
```

### 4) Extract Features Only (from canonical segment metadata)

```bash
uv --directory fungal-cv-qdrant run python -m src.experiments.feature_extraction.generate_features
```

### 5) Unified Upload (to Qdrant)

```bash
uv --directory fungal-cv-qdrant run python -m src.utils.upload_qdrant \
  --features-json ../Dataset/segmented_features.json \
  --metadata-json ../Dataset/prepared_segments_metadata.json \
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

## Autolab Multi-Agent System

Five-agent orchestration layer for autonomous iterative experimentation.

| Agent | Model | Role |
|-------|-------|------|
| `autolab` | BigBrain | Orchestrator — invoke this one |
| `researcher` | BigBrain | Literature scout → `research/paper-ideas.md` |
| `planner` | MidBrain | Queue coordinator → assigns `run_id` + branch |
| `worker` | MiniBrain | Isolated experiment runner via git worktree |
| `reporter` | MiniBrain | Status summarizer → F1 summary + staircase chart |

### Usage

```bash
# From monorepo root
opencode
# Prompt: "run one autoresearch pass on retrieval experiment"
```

### Research Notebook

```
research/
├── paper-ideas.md   # hypotheses (Researcher writes, Planner reads)
├── results.tsv      # run ledger (Worker appends)
├── do-not-repeat.md # tried-and-failed strategies (Planner writes)
└── papers/          # markitdown-converted PDFs (Researcher writes)
```

### Test

```bash
uv --directory repos/fungal-cv-qdrant run pytest tests/ -q
uv --directory repos/fungal-cv-qdrant run ruff check src/experiments/
```

## Notes

- src/main.py was removed by design.
- Qdrant named vectors must still match extractor names.
- New experiments should include program.md and a colocated check.py target.
- Source collections `Dataset/curated_primary/` and `Dataset/incoming_low_quality/` are canonical; legacy `Dataset/original/` and `Dataset/new_data/` still work as fallback.
- Segment metadata at `Dataset/prepared_segments_metadata.json` carries `segment_path` fields; consumers should use those instead of `Dataset/segmented_image/{id}.jpg`.
