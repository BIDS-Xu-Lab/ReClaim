#!/bin/bash
# Regenerate every RWE table and figure from the pipelines' results.
# Run from anywhere; outputs go to output/tables/ and output/figures/.
#   PYTHON         interpreter for the tables (any Python 3 with the standard library)
#   PYTHON_FIGURE  interpreter for the figures (needs matplotlib; 3.11.2 reproduces them byte for byte)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}
PYTHON_FIGURE=${PYTHON_FIGURE:-$PYTHON}
"$PYTHON" "$HERE/study1_tables.py"
"$PYTHON" "$HERE/study2_tables.py"
"$PYTHON_FIGURE" "$HERE/rwe_main_figure.py"
"$PYTHON_FIGURE" "$HERE/rwe_study1_doubly_robust.py"
