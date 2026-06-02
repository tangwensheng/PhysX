## DCU/HIP GPU architecture configuration
## Equivalent to SetCudaArch.cmake for NVIDIA builds
##
## Supports Hygon DCU architectures:
##   gfx936 — DCU K100 (CDNA2-like)
##   gfx938 — DCU K200 (CDNA3-like)

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

# Default DCU architectures (mirrors CUDA sm_70..sm_120 range)
if(NOT DEFINED DCU_ARCHITECTURES)
    set(DCU_ARCHITECTURES "gfx936,gfx938" CACHE STRING "DCU GPU architectures to target")
endif()
