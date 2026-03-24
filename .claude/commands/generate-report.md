Generate a structured research report for a phase of work, following the project's Definition of Done.

## Usage

Run this command and provide:
- **Phase/week folder** (e.g. `week_1_2`, `week_3_4`)
- Optional: a specific DOD file path (defaults to `report/<folder>/report_dod.md`)

## What the agent will do

1. Read the DOD from `report/<folder>/report_dod.md` or `report/<folder>/tasks.prompt.md`
2. Gather context: `git log`, changed files, results CSVs, existing visualizations
3. Write a complete report to `report/<folder>/REPORT.md` covering:
   - Overview (problem intro, glossary, purpose)
   - Methodology (fold design, split rationale, Mermaid diagrams)
   - Implementation (pipeline brief, script block diagram)
   - Results (comparison table, best config analysis, sensitivity)
   - Conclusion (recommendation)

## Invocation

Use the `research-report` skill to execute this workflow. When invoked, ask the user for the target folder if not already clear from context.

Example:
- "generate report for week_1_2" → writes `report/week_1_2/REPORT.md`
- "write the CV report" → infers `report/week_1_2/` from tasks.prompt.md
