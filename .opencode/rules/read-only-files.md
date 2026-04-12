# Rule: READ-ONLY Files

## Files marked with `# READ-ONLY` at the top MUST NOT be edited.

Any file that begins with the comment `# READ-ONLY` must not be modified by the agent under any circumstances. These files represent immutable contracts or reference data.

```python
# READ-ONLY
# This file is a generated contract. Do not edit manually.
```

## How to find READ-ONLY files:

```bash
grep -r "^# READ-ONLY" --include="*.py" src/
```

## If a READ-ONLY file needs to change:

1. Do NOT edit it directly.
2. Discuss with the user — the file may need to be regenerated from source, or a new experiment/contract file should be created instead.
3. The `# READ-ONLY` marker indicates the file is a machine-generated output or an immutable data contract.

## Current READ-ONLY files (if any):

Check with `grep -r "^# READ-ONLY" --include="*.py" src/`
