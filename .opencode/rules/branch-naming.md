# Rule: Experiment Branch Naming (autoresearch)

## Branch Format

All feature/experiment branches MUST follow this naming convention:

```
autoresearch/{experiment-name}/{N}-{summary}
```

### Components

| Component | Meaning |
|-----------|---------|
| `autoresearch/` | Prefix indicating this is an experiment branch |
| `{experiment-name}` | Short identifier for the experiment category (e.g., `segmentation`, `feature-extractor`, `embedding-lr`) |
| `{N}` | Sequential attempt number (1, 2, 3, ...) — increments each time you try something new |
| `{summary}` | Short kebab-case description of what changed in this attempt |

### Examples

```
autoresearch/segmentation/1-kmeans-baseline
autoresearch/segmentation/2-kmeans-n-clusters-5
autoresearch/feature-extractor/1-efficientnetb1-finetuned
autoresearch/feature-extractor/2-add-yolo-segmentation
autoresearch/embedding-lr/1-triplet-loss-baseline
```

## Merging Best Results

When an experiment run produces a **new best result** (higher accuracy than any previous attempt):

1. Merge the winning branch to `autoresearch/{experiment-name}` (no attempt suffix — this is the canonical best).
2. The `{experiment-name}` branch (e.g., `autoresearch/segmentation`) always holds the code that produced the best known result.
3. Discarded attempts (worse results) are NOT merged — they remain as historical record in their numbered branches.

## Naming Rules

- `{experiment-name}`: lowercase, hyphenated, max 3 words
- `{summary}`: lowercase, hyphenated, max 5 words, describes the key change
- Do NOT use spaces, underscores, or special characters in branch names
- The attempt number resets only when starting a brand new experiment category

## Creating a New Experiment

```bash
git checkout -b autoresearch/my-experiment/1-initial-baseline
```

Then implement the experiment in `src/experiments/<name>/`. See `src/run.py` for the experiment runner interface.
