---
name: research-report
description: "Generate or update experiment research reports using report/{exp_name}/{report_number}/content.md context, enforce nested report folder templates, and validate LaTeX with report/render_report.py. Use for 'generate report', 'write report', or LaTeX report rendering/validation tasks."
---

# Research Report Skill

## Use When

- User asks to generate/update report content for an experiment folder
- User asks to render report PDF from LaTeX
- User asks to validate report folder structure or LaTeX build

## Canonical Report Structure

- `report/{exp_name}/{report_number}/content.md` (context source)
- `report/{exp_name}/{report_number}/notes.txt` (free-form notes)
- `report/{exp_name}/{report_number}/results.txt` (result summary)
- `report/{exp_name}/{report_number}/images/` (visualization assets)
- `report/{exp_name}/{report_number}/main.tex` (LaTeX output)
- `report/templates/main.tex` (template source)

The renderer (`report/render_report.py`) bootstraps missing `content.md`, `notes.txt`, `results.txt`, `images/README.md`, and `main.tex`.

## Workflow

1. Resolve target experiment and report number (e.g. `cross_validation/001`).
2. Read context from `report/{exp_name}/{report_number}/content.md`.
3. Gather execution context: git history, changed files, result CSVs, report images.
4. Write/update `report/{exp_name}/{report_number}/main.tex`.
5. Validate and render with:

```bash
uv run python report/render_report.py --report-dir report/<exp_name>/<report_number>
```

6. If needed, skip LaTeX validation compile with:

```bash
uv run python report/render_report.py --report-dir report/<exp_name>/<report_number> --skip-validate
```

## LaTeX Validation Rules

- Default mode runs `pdflatex -interaction=nonstopmode -halt-on-error`.
- Any LaTeX error fails the command (non-zero exit).
- Fix invalid TeX and rerun validation before marking done.

## Report Writing Guidelines

- Follow `references/report_structure.md` for section templates.
- Include metrics/tables grounded in available artifacts.
- Keep figures and paths reproducible and workspace-relative.
