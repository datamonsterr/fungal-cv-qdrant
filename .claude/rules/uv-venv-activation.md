# Rule: uv and .venv Activation

## Always use `uv run` for all Python commands in this project.

```bash
# WRONG
python src/main.py
source .venv/bin/activate && python src/main.py

# CORRECT
uv run python src/main.py
```

## For interactive shell sessions (debugging, REPL):

```bash
source .venv/bin/activate          # bash/zsh
source .venv/bin/activate.fish    # fish shell
```

After activation, you can run `python` directly without `uv run`.

## When installing dependencies:

```bash
uv sync                           # install all dependencies from pyproject.toml
uv add <package>                  # add a dependency (updates both pyproject.toml and uv.lock)
uv add --dev <package>            # add a dev dependency
```

Then also update `requirements.txt` to keep it in sync:
```bash
uv pip freeze > requirements.txt
```

## Why

`uv` is the project's package manager. Using `uv run` ensures:
- The correct Python version from `.venv` is used
- All dependencies from `pyproject.toml` / `uv.lock` are available
- No accidental use of system Python or globally installed packages
