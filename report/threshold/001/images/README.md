# Images

## Confusion Matrices

- `cm_wtd_rat_halving_roc_opt.png` — Overall best strategy. F1=0.4587.
  TP=172, FP=368, TN=283, FN=38.
- `cm_spread_012_ov_s0_f1_grid.png` — Best f1_grid strategy. F1=0.4579.
  TP=166, FP=349, TN=302, FN=44.
- `cm_rat01_p_rat12_roc_opt.png` — Previous best (attempt 5). F1=0.4536.
  TP=166, FP=356, TN=295, FN=44.

## Methodology Diagram

- `methodology_diagram.mmd` — Mermaid source for the pipeline flowchart.
  Render with any Mermaid renderer or paste into mermaid.live.

## Generating diagrams

The confusion matrices were generated from `results/threshold/log/all_experiments.csv`
using Python + matplotlib.
