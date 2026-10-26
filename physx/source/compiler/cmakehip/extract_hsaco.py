#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
