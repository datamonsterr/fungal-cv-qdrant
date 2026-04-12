# Rule: Experiment Visualization — Staircase Chart

## Core Principle

Plot every individual experiment as a **single dot** on a staircase chart.
The x-axis represents the **experiment index** (not attempt number).
Each experiment's F1 score determines its height. A running best is tracked as we go through experiments in chronological order.

## Staircase Chart Rules

When plotting the autoresearch accuracy chart (`results/autoresearch/{experiment}.png`):

1. **X axis**: experiment index (0, 1, 2, ..., N-1) — each individual experiment gets one x position
2. **Gray dots**: experiments whose F1 falls **at or below** the current running best staircase line
3. **Green dots**: experiments that **set a new running best** at that point — these update the staircase
4. **Green staircase line**: horizontal-only segments (y increases, never decreases), connects every green dot in chronological order. A horizontal segment means the best stayed the same for that stretch.
5. **Label format**: on each green dot, show `{formula}_{algorithm}` — the strategy name that achieved the new best. Keep labels short (<25 chars).
6. **Legend**: two items only — "discarded" (gray dot, size 15) and "new best" (green circle, size 60)
7. **Y axis**: F1 score 0.0–1.0, formatted as decimal (0.00, 0.05, 0.10...)
8. **Y limit**: always starts at 0, upper bound = max_f1 × 1.15

## Implementation Details

### For threshold experiment (many experiments per attempt)

The threshold experiment generates **many experiments per attempt** (e.g., 162 formulas × 3 algorithms = 486 experiments per run). The chart accumulates ALL experiments across all attempts.

Each individual `formula × algorithm` pair is one dot. When a new best is set, the staircase steps up horizontally to the new F1 level.

```
experiment_index:  0   1   2   3   4   5   6   7   8   9  10  ...
f1:               0.2 0.3 0.3 0.1 0.2 0.4 0.3 0.2 0.4 0.3 0.5  ...
                     ●gray ●gray ●gray ●gray ●green→ ●gray ●green→ ●green
running_best:      0.2 0.3 0.3 0.3 0.3 0.4 0.4 0.4 0.4 0.4 0.5  ...
```

- Experiments 0, 1, 2, 3, 4: gray (below best)
- Experiment 5: GREEN (new best=0.4), staircase steps up from 0.3→0.4
- Experiment 6, 7: gray
- Experiment 8: GREEN (best=0.4, same level), horizontal segment continues
- Experiment 10: GREEN (new best=0.5), staircase steps up

### Per-experiment label guidelines

Each green dot is labeled with `{formula}_{algorithm}` (truncated to 20 chars).
Examples:
- `rat01_p_rat12_roc_opt` (26 chars → keep full, it's informative)
- `gnorm_0_2_f1_grid` (19 chars)
- `avg_top2_otsu` (14 chars)

## Do NOT

- Do NOT connect gray dots with any line
- Do NOT use diagonal segments in the staircase
- Do NOT show colored lines for multiple strategies on the same chart
- Do NOT auto-generate experiment names — agent specifies meaningful names in the description
- Do NOT plot attempt number on x-axis for threshold experiments — plot experiment index
