"""One-cell Colab bootstrap config.

Copy the string returned by get_colab_setup_cell() into a Colab cell and run it
before executing training scripts.
"""

from textwrap import dedent


def get_colab_setup_cell() -> str:
    return dedent(
        """
        !pip install -q uv
        !uv sync

        import os
        from pathlib import Path

        PROJECT_ROOT = Path('/content/fungal-cv-qdrant')
        os.chdir(PROJECT_ROOT)

        print('Project root:', PROJECT_ROOT)
        print('Run training modules like:')
        print('!uv run python -m src.experiments.finetune_dl.train_models')
        """
    ).strip()


if __name__ == "__main__":
    print(get_colab_setup_cell())
