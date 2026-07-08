#!/bin/bash
# Extract .hsaco files from .cu.o object files
OBJDIR="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"

find "$OBJDIR" \( -name '*.cu.o' -o -name '*.cu.cpp.o' \) | while read f; do
    base=$(basename "$f" | sed 's/\.cu\.o$//' | sed 's/\.cu\.cpp\.o$//')
    objcopy --dump-section .hip_fatbin="$OUTDIR/$base.hsaco" "$f" 2>/dev/null
    if [ -s "$OUTDIR/$base.hsaco" ]; then
        echo "$base.hsaco"
    else
        rm -f "$OUTDIR/$base.hsaco"
    fi
done
