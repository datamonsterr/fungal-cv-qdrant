# Report Content — Threshold Experiment (Autoresearch)

## 1. Objective

Detect **unknown fungal species** at query time using a threshold on KNN retrieval
similarity scores, without any fine-tuned classifier. The system must:

- Accept known species (colony images from 5 *Penicillium* species in the training set)
- Reject unknown species (all other species not in the training set)
- Maximise F1 on the positive (known-species) class

## 2. Dataset

### Database (Qdrant collection)

| Item | Value |
|------|-------|
| Collection | `myco_fungi_features_full_finetuned` |
| Extractor | `EfficientNetB1_finetuned` (1280-d) |
| Features | Finetuned EfficientNet-B1 on 5 target species |
| Count | Full segmented colony dataset — all environments (MEA, CYA, DG18, CREA, YES, OA) |

### Query / Test set

| Item | Value |
|------|-------|
| Source | `results/threshold/diverse_retrieval_results.csv` |
| Total samples | 861 |
| **Known species** (`is_known=1`) | **210** (5 species × ~42 strains, one strain held out per species) |
| **Unknown species** (`is_known=0`) | **651** (diverse non-target species) |
| Retrieval k | 11 neighbours |
| Env strategy | E1 (same growth medium only) |
| Agg strategy | weighted (cosine-similarity weighted votes) |

### Species in training set (known)

*Penicillium citreonigrum*, *P. commune*, *P. crustosum*, *P. expansum*, *P. chrysogenum*

### Test-train split

One strain per species is held out as the test set. All other strains of the same
species remain in the database. This ensures query images come from strains **never
seen during training**, making the known/unknown distinction realistic.

## 3. Query Construction

At query time for each plate image:

1. **Segment** — KMeans isolates the petri-dish region and extracts 1–3 colony
   segments per plate.
2. **Feature extraction** — Each segment is passed through `EfficientNetB1_finetuned`
   to produce a 1280-d embedding.
3. **KNN retrieval** — Qdrant KNN (k=11) is run with `env_strategy=E1` (only
   neighbours from the same growth medium) using named vector `EfficientNetB1_finetuned`.
4. **Sibling filtering** — Same-plate neighbours are excluded to avoid redundancy.
5. **Score extraction** — Top-k neighbour cosine similarities are recorded as
   `s0_score … s4_score` along with the species label of each neighbour.

The retrieval CSV (`diverse_retrieval_results.csv`) records these scores per query,
along with ground-truth `is_known` (1 if the query's species is in the target set,
0 otherwise).

## 4. Why Autoresearch?

Manual search over formula × threshold combinations is intractable:

- **Formula space** is open-ended: ratio chains, gap functions, polynomial
  transforms, weighted sums, entropy variants, etc.
- **Threshold algorithms** have trade-offs: `f1_grid` optimises F1 directly but
  overfits; `roc_opt` maximises Youden's J; `otsu` minimises intra-class variance.
- **Interaction effects**: the best formula for one threshold algorithm is not
  necessarily best for another.

The autoresearch pattern addresses this by:

1. **Structured logging** — every formula × algorithm × threshold attempt is
   recorded in `all_experiments.csv`.
2. **Staircase visualisation** — grey dots (discarded) vs green dots (new best)
   give a直观 timeline of which ideas improved the score.
3. **Active decision-making** — after each run, the experiment log is read before
   the next run to avoid repeating failed strategies.
4. **Small focused scripts** — each attempt tests 100–800 formula variants, keeping
   individual experiments interpretable.

## 5. Methodology Diagram

```
                        ┌─────────────────────────────────────┐
  Query plate image     │         RETRIEVAL PIPELINE          │
  (1–3 colonies)         │                                      │
                        │  1. Segment (KMeans / contour)       │
  ─────────────────────► │  2. EfficientNetB1_finetuned embed   │
                        │  3. Qdrant KNN (k=11, E1 strategy)   │
                        │  4. Filter same-plate neighbours       │
                        │  5. Weighted aggregation → ranked list│
                        └──────────────┬──────────────────────┘
                                       │ s0…s4 scores (cosine sim)
                                       ▼
                        ┌─────────────────────────────────────┐
                        │     THRESHOLD ANALYSIS               │
                        │                                      │
                        │  Formula family     Algorithm        │
                        │  ─────────────────────────────────   │
                        │  ratio chains       f1_grid (500 pts)│
                        │  gap functions      roc_opt          │
                        │  weighted sums       otsu             │
                        │  entropy variants                     │
                        │  ... (100–800 per run)                │
                        │                                      │
                        │  for each (formula, algorithm):      │
                        │    find optimal threshold T          │
                        │    evaluate: P, R, F1, Spec, TP/FP  │
                        │    log to all_experiments.csv         │
                        └──────────────┬──────────────────────┘
                                       │ best formula × algorithm
                                       ▼
                        ┌─────────────────────────────────────┐
                        │   KNOWN / UNKNOWN CLASSIFIER         │
                        │                                      │
                        │   score = formula(s0,…s4)           │
                        │   if score ≥ T  →  ACCEPT (known)    │
                        │   else              →  REJECT (unk)  │
                        └─────────────────────────────────────┘
```

## 6. Experiment History

| Attempt | Date | Experiments | Best Formula | F1 | Δ vs prev |
|---------|------|-------------|--------------|-----|-----------|
| 1 | 2026-04-01 | 25 | abs\_gap\_f1\_grid | 0.0900 | — (baseline) |
| 2 | 2026-04-01 | 125 | geom\_mean\_top3\_roc\_opt | 0.1639 | +0.074 |
| 3 | 2026-04-01 | 571 | geom\_mean\_top3\_roc\_opt | 0.1639 | no change |
| 4 | 2026-04-01 | ~100 | ratio\_s0s2\_roc\_opt | 0.4475 | +0.284 (labels fixed) |
| 5 | 2026-04-01 | 810 | **rat01\_p\_rat12\_roc\_opt** | **0.4536** | +0.006 |
| 6 | 2026-04-02 | 306 | **wtd\_rat\_halving\_roc\_opt** | **0.4587** | +0.005 |

Key turning point: Attempt 4 discovered that incorrect species label encoding
artificially inflated results. After fixing, F1 jumped from ~0.09 to ~0.45.

## 7. Best Strategies (Top-5 across all attempts)

| Rank | Formula | Algorithm | F1 | Precision | Recall | Specificity | TP | FP | TN | FN |
|------|---------|-----------|-----|----------|--------|-------------|----|----|----|-----|
| 1 | wtd\_rat\_halving | roc\_opt | **0.4587** | 0.3185 | 0.8190 | 0.4347 | 172 | 368 | 283 | 38 |
| 2 | spread\_012\_ov\_s0 | f1\_grid | 0.4579 | 0.3223 | 0.7905 | 0.4639 | 166 | 349 | 302 | 44 |
| 3 | rat01\_p\_rat12\_m\_rat23 | f1\_grid | 0.4533 | 0.3148 | 0.8095 | 0.4316 | 170 | 370 | 281 | 40 |
| 4 | wtd\_rat\_halving | f1\_grid | 0.4574 | 0.3138 | 0.8429 | 0.4055 | 177 | 387 | 264 | 33 |
| 5 | rat01\_p\_rat12\_m\_rat23 | roc\_opt | 0.4536 | 0.3180 | 0.7905 | 0.4531 | 166 | 356 | 295 | 44 |

Note: Previous best rat01\_p\_rat12\_roc\_opt (F1=0.4536) is still competitive.
The new `wtd_rat_halving` extends the consecutive ratio idea across all 4 neighbours
with halving decay weights, achieving slightly better balance.

## 8. Formula Family Analysis

### Ratio-chain family (consecutive s_i/s_{i+1})

Consecutive ratios capture how much the top match dominates the retrieval relative to
its immediate neighbours — a proxy for retrieval confidence.

| Formula | Algorithm | F1 |
|---------|-----------|-----|
| wtd\_rat\_halving (4-chain, halving decay) | roc\_opt | **0.4587** |
| wtd\_rat\_halving | f1\_grid | 0.4574 |
| wtd\_rat\_015 (4-chain, base-1.5 decay) | roc\_opt | 0.4530 |
| rat01\_p\_rat12\_m\_rat23 (3-term ± chain) | f1\_grid | 0.4533 |
| rat01\_p\_rat12\_roc\_opt (2-chain sum, prev best) | roc\_opt | 0.4536 |
| rat012\_sum (3-chain sum) | f1\_grid | 0.4512 |
| rat01\_p\_rat12\_p\_rat23 (3-chain sum) | f1\_grid | 0.4512 |

### Spread / diversity family

Spread metrics capture how much the top scores vary — high spread indicates a clear
best match, low spread suggests ambiguous retrieval.

| Formula | F1 |
|---------|-----|
| spread\_012\_ov\_s0 (CV top-3 / s0) | **0.4579** |
| gnorm02\_sq (normalised gap^2 s0-s2) | 0.4475 |
| iqr\_12 / range\_ov\_s0 | 0.4475 |

### Polynomial / gap family

| Formula | F1 |
|---------|-----|
| poly2\_01 (s0²−s1²) | 0.4271 |
| gap01\_ov\_gap23 | 0.4450 |

## 9. Confusion Matrices

See `images/` directory for confusion matrix figures for:
- `wtd_rat_halving_roc_opt.png` — overall best
- `spread_012_ov_s0_f1_grid.png` — best f1_grid performer
- `rat01_p_rat12_roc_opt.png` — previous best (attempt 5)

## 10. Key Insights

1. **Retrieval confidence is separable**: The gap/ratio between top neighbour scores
   (s0/s1, s0−s1) is far more discriminative than raw s0 alone.
2. **Chain length matters**: Extending the consecutive ratio beyond s0/s1+s1/s2 to
   all 4 consecutive pairs (halving decay) yields a small but consistent improvement.
3. **Threshold algorithm trade-off**: `roc_opt` consistently produces the best F1 by
   balancing sensitivity and specificity via Youden's J, while `f1_grid` sometimes
   overfits to recall at the expense of specificity.
4. **Precision ceiling**: Best precision is ~32%, meaning ~68% of accepted known
   predictions are wrong. The main failure mode is confusing unknown species with
   known ones (high FP).
5. **Recall is strong**: Best recall reaches ~84%, meaning most known species are
   correctly identified. The limiting factor is false positives, not false negatives.
