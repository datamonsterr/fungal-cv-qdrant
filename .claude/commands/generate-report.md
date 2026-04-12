Generate or update an experiment report via the `research-report` skill.

## Usage

Run this command and provide:
- **Experiment name** (e.g. `cross_validation`, `kmeans_segmentation`)
- **Report number** (e.g. `001`, `r1`, `2026-03-28`)
- Optional: a specific content context path (defaults to `report/<exp_name>/<report_number>/content.md`)

## What the agent will do

1. Ensure report structure exists under `report/<exp_name>/<report_number>/`:
	- `content.md`
	- `notes.txt`
	- `results.txt`
	- `images/README.md`
	- `main.tex`
2. Read context from `report/<exp_name>/<report_number>/content.md`
3. Gather context: `git log`, changed files, results CSVs, existing visualizations
4. Write or update `report/<exp_name>/<report_number>/main.tex`
5. Render and validate LaTeX with:
	`uv run python report/render_report.py --report-dir report/<exp_name>/<report_number>`

## Invocation

Use the `research-report` skill to execute this workflow. Ask for target folder if not clear.

Validation behavior:
- Default render validates LaTeX (`pdflatex -halt-on-error`)
- Optional fallback: `--skip-validate` if the user explicitly requests non-blocking render

Example:
- "generate report for cross_validation report 001" → updates `report/cross_validation/001/main.tex`
- "write segmentation report r2" → updates `report/kmeans_segmentation/r2/main.tex`
