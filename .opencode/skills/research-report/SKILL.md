---
name: research-report
description: "Generate or update experiment research reports with LaTeX PDF output and mermaid diagram rendering. Creates report/{exp_name}/{report_number}/ with main.tex, images/, and renders mermaid to PNG. Use for 'generate report', 'write report', 'compile report', or 'mermaid to pdf' tasks."
---

# Research Report Skill

## Use When

- Generate/update report content for an experiment folder
- Render report PDF from LaTeX (main.tex → main.pdf)
- Render mermaid diagrams to PNG images
- Validate report folder structure or LaTeX build

## Report Structure

```
report/{exp_name}/{report_number}/
├── content.md           # Context source
├── notes.txt            # Free-form notes
├── results.txt           # Result summary
├── images/              # Visualization assets
│   ├── diagram.png      # Rendered mermaid diagrams
│   └── *.png           # Other figures
├── main.tex            # LaTeX source
└── main.pdf            # Compiled PDF (output)
```

## Workflow

1. Resolve target: `report/<exp_name>/<report_number>/`
2. Read `content.md` for context and mermaid blocks
3. Create structure if missing (run `report/render_report.py`)
4. Write/update `main.tex`
5. Render mermaid diagrams to `images/` if present
6. Compile to PDF

## Mermaid → PNG Rendering

Extract mermaid blocks from content.md or create standalone:

```bash
# Using mermaid CLI (if installed)
mmdc -i diagram.mmd -o images/diagram.png

# Or use Python with mermaid library
python -m mermaid images/diagram.mmd images/diagram.png
```

Mermaid syntax rules:
- Use `flowchart` not `graph`
- Node IDs no spaces: `A[Node]` not `A [Node]`
- Subgraph names quoted: `subgraph A["Agent"]`

## Compile LaTeX to PDF

```bash
# Standard compile (creates main.pdf)
cd report/<exp_name>/<report_number> && pdflatex -interaction=nonstopmode main.tex

# Or use render script
uv run python report/render_report.py --report-dir report/<exp_name>/<report_number>
```

## Image Paths in LaTeX

Use relative paths from the report directory:

```latex
\includegraphics{images/figure.png}
\includegraphics{images/diagram.png}
```

## LaTeX Template

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{float}

\begin{document}
\title{Experiment Report}
\author{Myco Fungi Project}
\maketitle

\section*{Overview}
% from content.md

\section*{Results}
% from results.txt

\section*{Figures}
% \includegraphics{images/*.png}

\end{document}
```

## Validation

- `pdflatex -halt-on-error` (non-zero exit on error)
- Check PDF visually after compilation
- Fix TeX errors and rerun
