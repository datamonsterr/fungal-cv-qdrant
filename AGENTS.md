# Fungal CV — Autoresearch Agentic Loop

## Project Overview

**Task:** Myco fungi species classification from colony images.
**Target:** Classify 5 *Penicillium* species grown across multiple environments.

## Architecture

```
src/
├── experiments/{name}/     # Each experiment is self-contained
│   ├── program.md          # Agent instructions for this experiment
│   ├── prepare.py          # Immutable after tested — runs data + eval + visualize
│   ├── run_accuracy.py    # Agent modifies this to try new strategies
│   └── log/               # Structured log of all attempts
├── run.py                  # Main entry: runs prepare.py → records accuracy
├── prepare.py              # Prerequisite checks then run.py
└── lib/                    # Shared: cross_validation, metrics

results/autoresearch/{name}.csv   # Experiment history
results/autoresearch/{name}.png   # Staircase chart
```

## Experiment Workflow (autoresearch loop)

```
1. Agent modifies src/experiments/{name}/run_accuracy.py or related files
2. Run: uv run python src/prepare.py --experiment {name}
   → Checks: dataset, Qdrant, metadata
   → Runs: uv run python src/run.py --experiment {name} --description "what changed"
   → Output: F1 accuracy + staircase chart update
3. Agent reads log to decide next strategy
4. Repeat
```

## Commands

```bash
# Run experiment
uv run python src/prepare.py --experiment threshold

# List experiments
uv run python src/run.py --experiment-list

# Check prerequisites only
uv run python src/prepare.py --experiment threshold --skip-checks
```

## Accuracy Metrics

- **Primary:** F1 score (0.0–1.0)
- **Threshold experiment:** F1 for known vs unknown species separation
- **Retrieval experiments:** First-ranked species accuracy via 5-fold CV

## Staircase Chart Rules

- **Gray dots:** Discarded (worse than running best)
- **Green circles:** New best (kept checkpoint)
- **Green staircase:** Horizontal-only segments connecting green dots
- **Labels:** `{strategy}_{algorithm}` on each green dot (≤25 chars)

## Branch Naming

```
autoresearch/{experiment-name}/{N}-{summary}
```
Merge best results to `autoresearch/{experiment-name}` (no suffix).

## Key Files

| File | Purpose |
|------|---------|
| `src/experiments/{name}/program.md` | Agent instructions for that experiment |
| `src/experiments/{name}/prepare.py` | Immutable after created/tested |
| `src/experiments/{name}/run_accuracy.py` | Agent modifies this to try strategies |
| `results/autoresearch/{name}.csv` | Accuracy history (attempt, accuracy, kept, description) |
| `.claude/rules/experiment-visualization.md` | Full staircase chart specification |

## Environment Setup

```bash
uv sync && source .venv/bin/activate
docker compose up -d  # Qdrant required
```
