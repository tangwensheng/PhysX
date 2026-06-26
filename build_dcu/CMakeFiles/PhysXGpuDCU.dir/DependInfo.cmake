
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/public/home/tangwsh/PhysX/physx/source/gpubroadphase/src/CUDA/aggregate.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpubroadphase/src/CUDA/aggregate.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpubroadphase/src/CUDA/broadphase.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpubroadphase/src/CUDA/broadphase.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA/MemCopyBalanced.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA/MemCopyBalanced.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA/radixSortImpl.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA/radixSortImpl.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA/utility.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA/utility.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpunarrowphase/src/CUDA/cudaSphere.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpunarrowphase/src/CUDA/cudaSphere.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusimulationcontroller/src/CUDA/SDFConstruction.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpusimulationcontroller/src/CUDA/SDFConstruction.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusimulationcontroller/src/CUDA/algorithms.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpusimulationcontroller/src/CUDA/algorithms.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "PX_DCU_PORT"
  "PX_PHYSX_GPU_STATIC"
  "PX_SUPPORT_GPU_PHYSX"
  "PhysXGpuDCU_EXPORTS"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/public/home/tangwsh/PhysX/physx/include"
  "/public/home/tangwsh/PhysX/physx/include/geometry"
  "/public/home/tangwsh/PhysX/physx/source/common/include"
  "/public/home/tangwsh/PhysX/physx/source/common/src"
  "/public/home/tangwsh/PhysX/physx/source/foundation/include"
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/include"
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/src/CUDA"
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/src/DCU/stubs"
  "/public/home/tangwsh/PhysX/physx/source/gpunarrowphase/include"
  "/public/home/tangwsh/PhysX/physx/source/gpunarrowphase/src/CUDA"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/include"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA"
  "/public/home/tangwsh/PhysX/physx/source/gpusimulationcontroller/include"
  "/public/home/tangwsh/PhysX/physx/source/gpusimulationcontroller/src/CUDA"
  "/public/home/tangwsh/PhysX/physx/source/gpubroadphase/include"
  "/public/home/tangwsh/PhysX/physx/source/gpubroadphase/src/CUDA"
  "/public/home/tangwsh/PhysX/physx/source/gpuarticulation/include"
  "/public/home/tangwsh/PhysX/physx/source/gpuarticulation/src/CUDA"
  "/public/home/tangwsh/PhysX/physx/source/lowlevel/api/include"
  "/public/home/tangwsh/PhysX/physx/source/lowlevel/software/include"
  "/public/home/tangwsh/PhysX/physx/source/lowlevel/common/include"
  "/public/home/tangwsh/PhysX/physx/source/lowleveldynamics/include"
  "/public/home/tangwsh/PhysX/physx/source/lowleveldynamics/shared"
  "/public/home/tangwsh/PhysX/physx/source/lowlevelaabb/include"
  "/public/home/tangwsh/PhysX/physx/source/geomutils/src"
  "/public/home/tangwsh/PhysX/physx/source/geomutils/src/mesh"
  "/public/home/tangwsh/PhysX/physx/source/geomutils/include"
  "/public/home/tangwsh/PhysX/physx/source/cudamanager/include"
  "/public/home/tangwsh/PhysX/physx/source/simulationcontroller/include"
  "/public/home/tangwsh/PhysX/physx/source/scenequery/include"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/public/home/tangwsh/PhysX/physx/source/gpucommon/src/DCU/gpu_module_table.cpp" "CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpucommon/src/DCU/gpu_module_table.cpp.o" "gcc" "CMakeFiles/PhysXGpuDCU.dir/public/home/tangwsh/PhysX/physx/source/gpucommon/src/DCU/gpu_module_table.cpp.o.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
