---
name: Bug Report
about: Report a reproducibility issue or code error
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear description of the bug.

## Cell Number

Which notebook cell fails? (e.g., Cell 7D, Cell 10)

## Environment

- Colab runtime type: [ ] GPU (T4) / [ ] GPU (other) / [ ] CPU
- Python version:
- transformers version: (`pip show transformers`)
- accelerate version: (`pip show accelerate`)
- datasets version: (`pip show datasets`)

## Error Message

```
Paste the full error traceback here.
```

## Steps to Reproduce

1. Mount Drive
2. Run Cell 1
3. Restart runtime
4. Run Cell X
5. Error appears

## Expected vs Actual Behaviour

**Expected:** The cell should...  
**Actual:** Instead, it...

## Additional Context

- Did you restart the runtime after Cell 1? [ ] Yes / [ ] No
- Does Drive show existing result files? [ ] Yes / [ ] No
- Have you tried Runtime → Factory reset runtime? [ ] Yes / [ ] No
