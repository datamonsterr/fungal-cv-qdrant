"""One-cell Colab bootstrap config.

Copy the string returned by get_colab_setup_cell() into a Colab cell and run it
before executing training scripts.
"""

from textwrap import dedent


def get_colab_setup_cell() -> str:
    return dedent(
        """
        !curl -LsSf https://astral.sh/uv/install.sh | sh

        import os
        from pathlib import Path

        os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
        PROJECT_ROOT = Path('/content/fungal-cv-qdrant')
        os.chdir(PROJECT_ROOT)
        !uv sync

        print('Project root:', PROJECT_ROOT)
        print('Run training modules like:')
        print('!uv run python -m src.experiments.finetune_dl.train_models')
        """
    ).strip()


if __name__ == "__main__":
    print(get_colab_setup_cell())
