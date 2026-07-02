#!/usr/bin/env python3
"""Extract .hip_fatbin section from .o files into .hsaco code objects."""
import os, sys, subprocess

objdir = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
count = 0

for root, dirs, files in os.walk(objdir):
    for f in files:
        if not (f.endswith('.cu.o') or f.endswith('.cu.cpp.o')):
            continue
        objpath = os.path.join(root, f)
        base = os.path.basename(f)
        base = base.replace('.cu.cpp.o', '').replace('.cu.o', '')
        outpath = os.path.join(outdir, base + '.hsaco')
        subprocess.run(['objcopy', '--dump-section',
            f'.hip_fatbin={outpath}', objpath],
            capture_output=True)
        if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
            count += 1
        else:
            os.path.exists(outpath) and os.remove(outpath)

print(f'Extracted {count} .hsaco files')
