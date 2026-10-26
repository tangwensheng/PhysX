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

## HCU/HIP device architecture configuration
## Equivalent to SetCudaArch.cmake for NVIDIA builds
##
## Supports Hygon HCU architectures:
##   gfx936 — HCU K100 (CDNA2-like)
##   gfx938 — HCU K200 (CDNA3-like)

# Generate --offload-arch arguments for hipcc
# Usage: GENERATE_HIP_ARCH_LIST(TARGETS "gfx936;gfx938")
macro(GENERATE_HIP_ARCH_LIST)
    cmake_parse_arguments(GENERATE_HIP_ARCH_LIST "" "TARGETS" "" ${ARGN})

    set(HIP_ARCH_FLAGS "")

    if (GENERATE_HIP_ARCH_LIST_TARGETS)
        string(REPLACE "," ";" hip_archs "${GENERATE_HIP_ARCH_LIST_TARGETS}")
        foreach (arch IN LISTS hip_archs)
            set(HIP_ARCH_FLAGS "${HIP_ARCH_FLAGS} --offload-arch=${arch}")
        endforeach ()
    else()
        # Default to gfx936 if not specified
        set(HIP_ARCH_FLAGS "--offload-arch=gfx936")
    endif ()

    set(HIP_ARCH_FLAGS ${HIP_ARCH_FLAGS} CACHE INTERNAL "HIP offload architecture flags")
endmacro()

# Default HCU architectures (mirrors CUDA sm_70..sm_120 range)
if(NOT DEFINED DCU_ARCHITECTURES)
    set(DCU_ARCHITECTURES "gfx936,gfx938" CACHE STRING "HCU device architectures to target")
endif()
