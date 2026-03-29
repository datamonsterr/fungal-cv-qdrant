---
name: create-new-experiment
description: "Bootstrap a new autoresearch experiment folder with run entrypoint, program.md, check contract, and report skeleton. Use when user asks to create or initialize a new experiment."
version: 1.0.0
author: project
---

# Create New Experiment

## Use When

- User asks to create a new experiment
- User asks to add a new research track or use case
- User asks for a reproducible experiment scaffold

## What to Create

1. src/experiments/<experiment_name>/
1. src/experiments/<experiment_name>/program.md
1. src/experiments/<experiment_name>/run.py
1. src/check/<experiment_name>_check.py
1. report/<experiment_name>/content.md
1. report/<experiment_name>/main.tex

## Requirements

- program.md must define objective, inputs, outputs, and run command.
- check file must define concise immutable target thresholds.
- run.py must expose a main() and be runnable with python -m.
- report folder must contain content.md for context and a LaTeX entrypoint.

## Suggested run.py skeleton

```python
import argparse


def run() -> None:
    print("Experiment placeholder")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
```
