#!/bin/bash
# Build the native rainflow C extension.
# Run from any directory — the script finds its own location.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/_rainflow.c"
OUT="$SCRIPT_DIR/_rainflow.so"

set -e
echo "Compiling $SRC → $OUT"
gcc -O3 -march=native -shared -fPIC -o "$OUT" "$SRC" -lm
echo "Done — $(ls -lh "$OUT" | awk '{print $5}') shared library ready."
