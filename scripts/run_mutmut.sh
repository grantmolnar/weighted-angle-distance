#!/usr/bin/env bash
set -euo pipefail

# Clean run
rm -rf mutants .mutmut-cache || true

# Baseline: tests must be green first
python -m pytest -q

# Fast iteration: you can pass wildcards to focus mutation on a module/function
# Examples from mutmut docs: mutmut run "my_module*"  :contentReference[oaicite:2]{index=2}
#
# Full project mutation run:
mutmut run

echo
echo "Mutation run complete."
echo "Next step: inspect survivors with: mutmut browse"

