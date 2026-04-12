import argparse
import shutil
import subprocess
from pathlib import Path


TEMPLATE = Path("report/templates/main.tex")
DEFAULT_CONTENT = """# Report Content

Use this file as context for report generation.

- Objective:
- Dataset version:
- Key metrics:
- Figures:
"""
DEFAULT_NOTES = """Report Notes

- Hypothesis:
- Observations:
- Risks:
- Next steps:
"""
DEFAULT_RESULTS = """Report Results

- Primary metric:
- Best configuration:
- Comparison summary:
- Artifact paths:
"""
DEFAULT_IMAGES_README = """# Images

Place visualization images for this report here.

Examples:
- confusion_matrix.png
- accuracy_vs_k.png
- env_comparison.png
"""
TEXT_FILES = {
    "content.md": DEFAULT_CONTENT,
    "notes.txt": DEFAULT_NOTES,
    "results.txt": DEFAULT_RESULTS,
}


def ensure_report_structure(report_dir: Path) -> Path:
    """Create required report files if they do not already exist.

    Required structure:
    - report/{exp_name}/{report_number}/content.md
    - report/{exp_name}/{report_number}/notes.txt
    - report/{exp_name}/{report_number}/results.txt
    - report/{exp_name}/{report_number}/images/
    - report/{exp_name}/{report_number}/main.tex
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    for file_name, file_content in TEXT_FILES.items():
        file_path = report_dir / file_name
        if not file_path.exists():
            file_path.write_text(file_content)

    images_dir = report_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_readme = images_dir / "README.md"
    if not images_readme.exists():
        images_readme.write_text(DEFAULT_IMAGES_README)

    tex_path = report_dir / "main.tex"
    if not tex_path.exists():
        shutil.copyfile(TEMPLATE, tex_path)

    return tex_path


def validate_latex(report_dir: Path, tex_name: str = "main.tex") -> None:
    """Compile LaTeX with strict error handling.

    Uses ``-halt-on-error`` so the command fails on invalid LaTeX.
    """
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_name,
        ],
        cwd=report_dir,
        check=True,
    )


def render(report_dir: Path, validate: bool = True) -> None:
    tex_path = ensure_report_structure(report_dir)

    if validate:
        validate_latex(report_dir, tex_name=tex_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render experiment LaTeX report")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip pdflatex validation compile",
    )
    args = parser.parse_args()
    render(Path(args.report_dir), validate=not args.skip_validate)


if __name__ == "__main__":
    main()
