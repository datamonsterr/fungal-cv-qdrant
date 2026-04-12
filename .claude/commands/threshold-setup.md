Reset and set up the threshold experiment on the refreshed segmented diverse retrieval pipeline.

## Mission

Build a new canonical threshold input from segmented diverse colony queries, then establish a fresh baseline F1 that every later threshold autoresearch attempt must build on.

## Hard reset

Do **not** trust old threshold artifacts produced from:

- full-plate `preprocessed` diverse queries
- stale `results/threshold/diverse_retrieval_results.csv`
- `src/experiments/threshold/retrieve_with_train_filter.py`
- any threshold analysis run created before the segmented diverse test-set retrieval refresh

The canonical threshold input must now come from:

- segmented diverse query images in `Dataset/diverse_data/.../*_seg{n}.jpg`
- grouped per-strain test sets that match the segmented retrieval and cross-validation logic
- same-environment retrieval against `myco_fungi_features_full_finetuned`
- weighted aggregation from segmented DB neighbors
- full visualization and JSON for every retrieved test set

## Known and Unknown data

Known samples:

- the 7 held-out Penicillium test strains prepared by `src/experiments/threshold/prepare_test_strains.py`

Unknown samples:

- all remaining diverse strains in `Dataset/diverse_data`

Rows in the refreshed retrieval CSV must be **per test set**, not per original image and not per whole strain.
One strain can contribute up to 6 test sets:

- `seg0_ob`
- `seg0_rev`
- `seg1_ob`
- `seg1_rev`
- `seg2_ob`
- `seg2_rev`

Each test set should contain one segmented query colony per available environment.

## Prerequisites

```bash
docker compose up -d
```

## Step 1 — Prepare held-out known test strains

```bash
uv run python -m src.experiments.threshold.prepare_test_strains
```

Expected outputs:

- `results/threshold/test_strain_retrieval_list.json`
- `Dataset/diverse_data/diverse_data_with_test_strains_metadata.json`

Important check:

- if the retrieval path still ignores the prepared known test strains or still reads the wrong metadata source, fix that first before trusting the next steps

## Step 2 — Rebuild segmented diverse retrieval with full visualization

Canonical retrieval implementation:

- `src/experiments/threshold/retrieve_diverse.py`

Requirements for this retrieval step:

- query `step_images["segments"]`, not full-plate `preprocessed` images
- build per-strain test sets using the same segmented test-set logic as the retrieval and cross-validation code
- skip diverse environments that do not exist in the DB
- use same-environment retrieval only
- regenerate the canonical `s0_score..s4_score` inputs from this refreshed retrieval
- write full visualization and JSON for every retrieved test set

If the current code path still mixes full images with segmented images, or does not include the prepared known test strains, fix that first before continuing.

```bash
uv run python -m src.experiments.threshold.retrieve_diverse

# Resume if interrupted:
uv run python -m src.experiments.threshold.retrieve_diverse --resume
```

Expected outputs:

- `results/threshold/diverse_retrieval_results.csv`
- `results/threshold/diverse_retrieval_visualizations/`
- `results/threshold/diverse_retrieval_json/`

## Step 3 — Validate the refreshed retrieval before threshold analysis

Do **not** continue until all of these are true:

1. `results/threshold/diverse_retrieval_results.csv` contains both `is_known=1` and `is_known=0`
2. each row is a test set such as `strain__set1`
3. `image_path` entries point to segmented diverse query paths such as `*_seg0.jpg`
4. retrieval visualizations show segmented diverse query crops and DB segmented neighbors
5. the score columns `s0_score..s4_score` come from the refreshed segmented retrieval, not from a stale run
6. the retrieval JSON files contain per-test-set `raw_results` with one query segment per available environment

If any check fails, fix the retrieval implementation and regenerate the retrieval outputs before running threshold analysis.

## Step 4 — Run the fresh baseline threshold experiment

Use the normal experiment entry point so the result is recorded in autoresearch history.

```bash
uv run python src/prepare.py --experiment threshold --description "baseline refreshed segmented diverse retrieval"
```

This run establishes the new baseline F1.
If the threshold experiment code still assumes stale retrieval semantics, fix that first and re-run the baseline.

Expected outputs:

- `results/threshold/threshold_analysis.csv`
- `results/threshold/all_strategy_results.csv`
- `results/autoresearch/threshold.csv`
- `results/autoresearch/threshold.png`

## Step 5 — Interpret the baseline with the staircase rule

Use `.claude/rules/experiment-visualization.md` as the source of truth for interpreting the staircase in:

- `results/autoresearch/threshold.csv`
- `results/autoresearch/threshold.png`

After the baseline run:

- identify the best current `{formula}_{algorithm}`
- identify whether it is the latest running best point on the staircase
- use that formula family as the center of gravity for the next experiment loop

Do **not** append logs to `.claude/rules/experiment-visualization.md`.
Use that file as the staircase specification, and log experiment outcomes in the threshold result and log files.

## Step 6 — Prepare the next session handoff

Before handing off, make sure these are current:

- `results/threshold/diverse_retrieval_results.csv`
- `results/threshold/diverse_retrieval_visualizations/`
- `results/threshold/diverse_retrieval_json/`
- `results/threshold/log/experiments.log`
- `results/threshold/log/best_strategy.json`
- `results/autoresearch/threshold.csv`
- `results/autoresearch/threshold.png`

The next autoresearch session should continue with `/threshold-experiment`.

## Key files

- `src/experiments/threshold/prepare_test_strains.py`
- `src/experiments/threshold/retrieve_diverse.py`
- `src/experiments/threshold/threshold_analysis.py`
- `src/experiments/threshold/expanded_threshold_analysis.py`
- `src/experiments/threshold/run.py`
- `src/analysis/visualization/visualize_prediction.py`
- `.claude/rules/experiment-visualization.md`
