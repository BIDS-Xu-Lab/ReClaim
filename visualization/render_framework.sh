#!/bin/bash
# Render the manuscript framework PDF for the repository README (requires Poppler).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pdftoppm -singlefile -png -scale-to 2400 \
    "$HERE/../figures/framework.pdf" "$HERE/../figures/framework"
