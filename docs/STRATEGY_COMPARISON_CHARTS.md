# Strategy Comparison Charts: Complete Guide

## Overview

The strategy comparison visualization (`ensemble_strategy_comparison.png`) consists of **4 charts in a 2×2 grid** that provide comprehensive analysis of how different ensemble strategies perform. This document explains each chart's purpose, interpretation, and how to use them for tuning.

---

## Chart 1: Overall Accuracy Comparison (Top-Left)

### Purpose
Compares the **overall accuracy** of different ensemble strategies against the best individual model.

### What It Shows
- **Bar Chart** displaying accuracy percentages
- **Weighted Sum Strategy** (cyan/teal `#4ecdc4`)
- **Simple Average Strategy** (blue `#45b7d1`)
- **Manual Weighted Strategy** (purple `#a29bfe`) - *if available*
- **Best Individual Model** (red `#ff6b6b`) - highest performing single feature extractor

### Interpretation

#### Reading the Chart
- **Y-axis**: Accuracy percentage (0-100%)
- **Bars**: Each bar shows overall prediction accuracy
- **Numbers on top**: Exact accuracy percentage

#### What to Look For
1. **Ensemble vs Individual Performance**
   - Ensemble should ideally outperform individual models
   - If Best Individual > Ensemble → models may not be complementary

2. **Strategy Differences**
   - Weighted vs Simple: Shows if accuracy-based weighting helps
   - Manual vs Others: Shows if species-specific tuning improves performance

3. **Performance Gaps**
   - Large gaps suggest significant strategy differences
   - Small gaps indicate similar behavior

### Example Insights
```text
Weighted Sum:    66.67%  ← Good performance
Simple Average:  66.67%  ← Same as weighted
Manual Weighted: 64.58%  ← Needs tuning! Lower than others
Best Individual: 75.00%  ← ColorHistogramHS performs better alone
```

**Action**: If manual weighted underperforms, adjust species-specific weights in `species_weights.json`.

---

## Chart 2: Prediction Agreement Patterns (Top-Right)

### Purpose
Analyzes **how often strategies agree or disagree** on predictions, revealing complementarity.

### What It Shows

#### Two-Way Mode (no manual weights)
- **Both Correct**: All strategies predict correctly
- **Weighted Only**: Only weighted strategy correct
- **Simple Only**: Only simple average correct
- **Both Wrong**: All strategies predict incorrectly

#### Three-Way Mode (with manual weights)
- **All 3 Correct**: All strategies predict correctly
- **2/3 Correct**: Two strategies agree on correct answer
- **Manual Only**: Only manual weighted correct
- **Weighted Only**: Only weighted sum correct
- **Simple Only**: Only simple average correct
- **All 3 Wrong**: All strategies predict incorrectly

### Interpretation

#### Reading the Chart
- **Y-axis**: Number of predictions (count)
- **Bars**: Frequency of each agreement pattern
- **Numbers on top**: Exact count

#### What to Look For
1. **High Agreement (All Correct/Wrong)**
   - Many "All Correct" → Good consensus on easy cases
   - Many "All Wrong" → Shared blind spots

2. **Disagreement Patterns**
   - "X Only" bars show unique contributions
   - Balanced disagreement suggests complementarity
   - Imbalanced suggests one strategy dominates

3. **Correction Opportunities**
   - "2/3 Correct" shows potential for improvement
   - If one strategy consistently wins alone → investigate why

### Example Insights
```
Three-Way Mode:
All 3 Correct:   28  ← Strong consensus on easy cases
2/3 Correct:     15  ← Potential for better combination
Manual Only:      3  ← Manual catches some unique cases
Weighted Only:    0  ← Weighted doesn't add unique value
Simple Only:      0  ← Simple doesn't add unique value
All 3 Wrong:      2  ← Shared difficult cases
```

**Action**: If one strategy never wins alone ("X Only" = 0), it's redundant. If "All Wrong" is high, all models struggle with certain species.

---

## Chart 3: Confidence Distribution (Bottom-Left)

### Purpose
Compares **prediction confidence distributions** across strategies using box plots.

### What It Shows
- **Box plots** for each strategy
- **Red median line**: Middle value (50th percentile)
- **Box**: Interquartile range (25th to 75th percentile)
- **Whiskers**: Min/max within 1.5×IQR
- **Circles**: Outliers beyond whiskers
- **μ annotation**: Mean confidence value

### Interpretation

#### Reading the Chart
- **Y-axis**: Prediction confidence (0.0 to 1.0)
- **Each box plot**: Distribution for one strategy
- **Colors**: 
  - Cyan: Weighted Sum
  - Blue: Simple Average
  - Purple: Manual Weighted (if present)

#### What to Look For
1. **Median Confidence (red line)**
   - Higher median → More confident predictions
   - Similar medians → Strategies have similar certainty

2. **Box Height (IQR)**
   - Tall box → High variability in confidence
   - Short box → Consistent confidence levels

3. **Mean vs Median (μ vs red line)**
   - Mean > Median → Right-skewed (some very high confidences)
   - Mean < Median → Left-skewed (some very low confidences)

4. **Outliers**
   - Many low outliers → Strategy uncertain on some cases
   - Many high outliers → Some predictions very confident

### Example Insights
```
Weighted:  μ=0.678, median=0.682
Simple:    μ=0.680, median=0.685
Manual:    μ=0.652, median=0.661

Observation: Manual weighted has lower confidence overall
Action: Check if lower confidence correlates with incorrect predictions
```

**Calibration Check**: Compare confidence to accuracy
- High confidence + Low accuracy → Overconfident, poorly calibrated
- Low confidence + High accuracy → Underconfident, but accurate
- Matched confidence/accuracy → Well-calibrated predictions

---

## Chart 4: Confusion Analysis - Different Predictions (Bottom-Right)

### Purpose
Analyzes cases where **strategies disagree** to identify which performs best on difficult examples.

### What It Shows

#### Three-Way Mode (with manual weights)
- **All Wrong**: All strategies failed (red `#ff6b6b`)
- **Weighted Best**: Only weighted sum correct (cyan `#4ecdc4`)
- **Simple Best**: Only simple average correct (blue `#45b7d1`)
- **Manual Best**: Only manual weighted correct (purple `#a29bfe`)
- **2+ Correct**: Two or more strategies correct (green `#00b894`)

#### Two-Way Mode (without manual weights)
- **Both Wrong (Different Pred)**: Both failed with different predictions
- **Weighted Better**: Only weighted sum correct
- **Simple Better**: Only simple average correct
- **Both Correct (Different Pred)**: Both correct with different predictions

### Interpretation

#### Reading the Chart
- **Y-axis**: Number of cases (count)
- **Title**: Shows total number of different predictions analyzed
- **Bars**: Frequency of each outcome pattern

#### What to Look For
1. **All Wrong Cases**
   - High count → Many shared difficult cases
   - Suggests need for different feature extractors or models

2. **X Best Cases**
   - Shows unique strengths of each strategy
   - If balanced → strategies are complementary
   - If imbalanced → one strategy dominates disagreements

3. **2+ Correct Cases**
   - High count → Good redundancy, multiple strategies catch errors
   - Low count → Strategies rarely agree on corrections

4. **Zero Values**
   - If "Weighted Best" = 0 → Weighted never uniquely correct
   - If "Manual Best" = 0 → Manual weights not helping

### Example Insights
```
Analysis of 16 Different Predictions:
All Wrong:        5  ← Difficult cases for all strategies
Weighted Best:    2  ← Weighted catches some unique cases
Simple Best:      1  ← Simple occasionally better
Manual Best:      3  ← Manual catches different cases than others
2+ Correct:       5  ← Good redundancy

Interpretation:
- Manual weighted adds value (3 unique corrections)
- All strategies struggle with 5 cases (31% of disagreements)
- Redundancy is good (5 cases with multiple correct predictions)
```

**Action**: If "All Wrong" is high, investigate those specific cases to understand shared blind spots.

---

## Using Charts Together for Tuning

### Workflow for Manual Weight Tuning

1. **Start with Chart 1**: Check if manual weighted accuracy is competitive
   - If much lower → weights need significant adjustment
   - If close → fine-tuning may help

2. **Check Chart 2**: Look for "Manual Only" bars
   - If 0 → Manual weights not adding unique value
   - If high → Manual catching cases others miss

3. **Review Chart 3**: Compare confidence distributions
   - Lower mean/median → May need to boost certain models
   - High variability → Inconsistent weight performance

4. **Deep dive Chart 4**: Identify unique contributions
   - "Manual Best" count shows unique corrections
   - "All Wrong" shows shared challenges

### Example Tuning Scenario

**Problem**: Manual weighted accuracy is 64.58%, lower than 66.67%

**Analysis**:
1. Chart 1: Confirms manual underperforms by ~2%
2. Chart 2: "Manual Only" = 3 cases → Manual adds some unique value
3. Chart 3: μ=0.652 vs 0.678 → Lower confidence overall
4. Chart 4: "Manual Best" = 3, "All Wrong" = 5 → Manual helps on specific cases

**Diagnosis**: Manual weights are catching unique cases but losing more on others

**Solution**:
1. Check `species_weights.json` for species with "All Wrong" cases
2. Increase weights for models that performed well on those species
3. Reduce weights for underperforming models on specific species
4. Re-run and compare all 4 charts again

---

## Chart Color Coding Reference

| Color | Hex Code | Strategy |
|-------|----------|----------|
| Cyan/Teal | `#4ecdc4` | Weighted Sum |
| Blue | `#45b7d1` | Simple Average |
| Purple | `#a29bfe` | Manual Weighted |
| Red | `#ff6b6b` | Best Individual / All Wrong |
| Green | `#00b894` | 2+ Correct |

---

## Quick Reference: When to Use Each Chart

| Question | Chart to Check |
|----------|---------------|
| "Is my ensemble better than individual models?" | **Chart 1** - Overall Accuracy |
| "Do my strategies complement each other?" | **Chart 2** - Agreement Patterns |
| "Are my strategies confident in predictions?" | **Chart 3** - Confidence Distribution |
| "Which strategy is best on difficult cases?" | **Chart 4** - Confusion Analysis |
| "Is manual weighted adding value?" | **All 4 charts** - Look for unique contributions |
| "Where should I focus tuning efforts?" | **Chart 4** - Check "All Wrong" cases |

---

## Advanced Interpretation

### Identifying Overconfidence
- **Chart 3**: High mean confidence (μ > 0.8)
- **Chart 1**: Low accuracy (< 70%)
- **Action**: Models may need calibration or re-weighting

### Detecting Redundancy
- **Chart 2**: "X Only" = 0 for a strategy
- **Chart 4**: "X Best" = 0 for a strategy
- **Action**: That strategy is not adding unique value

### Finding Complementary Models
- **Chart 2**: Balanced "X Only" counts
- **Chart 4**: All strategies have some "X Best" cases
- **Ideal**: Each strategy catches different errors

### Tuning Manual Weights
1. Identify species in "All Wrong" cases (check JSON output)
2. Look up those species in `complementary_cases_list.json`
3. Check which individual models performed well
4. Increase those model weights in `species_weights.json`
5. Re-run `ensemble_analysis.py` and compare charts

---

## Related Documentation

- **MANUAL_WEIGHTED_STRATEGY.md**: Comprehensive guide to manual weighting
- **MANUAL_WEIGHTED_IMPLEMENTATION.md**: Technical implementation details
- **ENSEMBLE_WEIGHT_EXPLANATION.md**: How weights are normalized
- **species_weights.json**: Configuration file for manual weights

---

## Notes

- Charts automatically adapt to available strategies (2-way vs 3-way comparison)
- If manual weights not loaded, charts show only weighted and simple average
- All charts use consistent color coding for easy cross-reference
- Generated by `ensemble_analysis.py` in the `create_strategy_comparison_chart()` function
