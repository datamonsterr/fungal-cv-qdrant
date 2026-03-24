Run all linters and type checks on the src/ directory.

```bash
uv run black src && uv run isort src && uv run flake8 src && uv run mypy src
```

Report any errors found. If all pass, confirm the code is clean.
