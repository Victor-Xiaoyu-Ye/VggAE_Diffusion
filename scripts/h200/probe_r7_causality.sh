#!/bin/bash
# Assert R7 temporal shape, gradient, and causal invariants.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" "${PROJECT}/probe_r7_causality.py" --factors 2 4
