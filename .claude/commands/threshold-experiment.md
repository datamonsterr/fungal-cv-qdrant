Run one new autoresearch attempt for the threshold experiment on top of the refreshed segmented diverse retrieval baseline.

Assume `/threshold-setup` has already been completed successfully.
If the retrieval CSV is stale, incomplete, or still based on the old retrieval path, run `/threshold-setup` first.

## Mission

Improve threshold F1 by changing methodology and implementation, not by changing the scoring target.
The refreshed segmented diverse retrieval CSV is the canonical source for `s0_score..s4_score`.

## Do not change

- the known-vs-unknown label meaning
- the primary experiment score definition
- the staircase interpretation rules in `.claude/rules/experiment-visualization.md`

## Allowed directions

- retrieval implementation fixes
- known-test-strain inclusion fixes
- segmented query grouping fixes
- environment handling fixes
- neighbor filtering fixes
- formula generation and threshold methodology improvements
- logging improvements
- threshold-aware visualization improvements
- any other implementation change that can improve F1 without changing the score definition

## Workflow for each new attempt

1. Read the current state before making changes.

   Review:

   - `results/threshold/diverse_retrieval_results.csv`
   - `results/threshold/log/experiments.log`
   - `results/threshold/log/best_strategy.json`
   - `results/threshold/log/all_experiments.csv`
   - `results/autoresearch/threshold.csv`
   - `results/autoresearch/threshold.png`

   Confirm the retrieval CSV is still the refreshed segmented test-set version before trusting any threshold result.

2. Create a new attempt branch.

   ```bash
   git checkout -b autoresearch/threshold/{N}-{summary}
   ```

3. Pick exactly one new methodology family.

   Use the latest staircase running best as the center of gravity.
   Fix implementation issues exposed by previous logs before inventing new formulas blindly.

   Good directions:

   - fix stale retrieval assumptions
   - fix known-test-strain inclusion bugs
   - improve segmented test-set construction
   - improve formula families around the current best new-best staircase point
   - improve threshold calibration implementation
   - improve threshold-aware visualization for failure analysis

   Bad directions:

   - repeating a previously logged method
   - changing the experiment score definition
   - relying on stale retrieval artifacts

4. Log the planned method before editing code.

   Record enough detail in `results/threshold/log/experiments.log` so the same attempt is not repeated later.
   If needed, also create or update an `attempt_XXX.json` note.

5. Make one focused code change.

   Most likely files:

   - `src/experiments/threshold/retrieve_diverse.py`
   - `src/experiments/threshold/threshold_analysis.py`
   - `src/experiments/threshold/expanded_threshold_analysis.py`
   - `src/experiments/threshold/run.py`
   - `src/analysis/visualization/visualize_prediction.py`

6. Rebuild retrieval whenever the attempt touches query construction or score inputs.

   Re-run these when the attempt changes any of the following:

   - query source data
   - known-test-strain inclusion
   - segmented test-set construction
   - environment filtering
   - neighbor filtering
   - any code path that changes `s0_score..s4_score`

   ```bash
   uv run python -m src.experiments.threshold.prepare_test_strains
   uv run python -m src.experiments.threshold.retrieve_diverse
   ```

   Do **not** continue until the retrieval checks from `/threshold-setup` pass again.

7. Run the experiment through the normal prepare/eval entry point.

   ```bash
   uv run python src/prepare.py --experiment threshold --description "what changed"
   ```

   If the attempt uses a different threshold-analysis path, wire it through the threshold experiment entry point so the final evaluation still runs through `src/prepare.py --experiment threshold`.

8. Evaluate the result with the staircase rule.

   Use `.claude/rules/experiment-visualization.md` as the staircase specification when reading:

   - `results/autoresearch/threshold.csv`
   - `results/autoresearch/threshold.png`

   After the run:

   - identify the newest best `{formula}_{algorithm}`, if one exists
   - compare it against prior running-best points in `results/autoresearch/threshold.csv`
   - use that winning formula family as the next place to dig deeper

9. After each newest best, generate threshold-aware prediction visualizations.

   Use `src/analysis/visualization/visualize_prediction.py`, but do **not** treat the top-1 species as the final experiment prediction.

   The final decision shown in the visualization must be the thresholded `known` or `unknown` output from the winning formula.

   Each new-best visualization must show:

   - formula name
   - algorithm name
   - threshold value
   - formula score for the test set
   - binary prediction: `known` or `unknown`
   - ground-truth binary label
   - ranked neighbor species as supporting context only
   - segmented diverse query images and DB segmented neighbors

10. Log the outcome immediately after the run.

    Update:

    - `results/threshold/log/experiments.log`
    - `results/threshold/log/best_strategy.json` if a new best was found
    - any per-attempt note file used during the run

    The log must clearly state:

    - what was tried
    - whether retrieval was regenerated
    - the resulting best F1
    - whether it became a new running best
    - what formula family should be investigated next

11. If the attempt is a new running best, merge it to the canonical best branch.

    ```bash
    git checkout autoresearch/threshold
    git merge autoresearch/threshold/{N}-{summary}
    ```

    If it is not a new best, keep the branch as historical record.

## Loop rule

Repeat the cycle:

1. inspect prior bests and prior failures
2. fix the implementation based on the latest logged findings
3. re-run `src/prepare.py --experiment threshold`
4. check the staircase for a new best
5. if there is a new best, generate threshold-aware visualizations and dig deeper in that formula family
6. log the method so it is not repeated

## Suggested search strategy

Use the previous logged best as the center of gravity.
Do not search blindly.

For each loop:

- inspect the current best formula family
- identify the likely implementation or methodological weakness around it
- make one focused improvement
- rerun the canonical threshold path
- keep only improvements that move the running best upward

## Parallel exploration

If you split work across subagents, use at most 4 independent methodology families in parallel.
Each must have:

- its own branch or worktree
- a distinct logged idea
- no duplication of a previously tried method

Merge only the winning branch back to `autoresearch/threshold`.
