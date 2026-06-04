
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/accumulateThresholdStream.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/accumulateThresholdStream.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/artiConstraintPrep2.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/artiConstraintPrep2.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrep.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrep.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrepTGS.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/constraintBlockPrepTGS.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/integration.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/integration.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/integrationTGS.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/integrationTGS.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/preIntegration.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/preIntegration.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/preIntegrationTGS.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/preIntegrationTGS.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solver.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solver.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solverMultiBlock.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solverMultiBlock.cu.o"
  "/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu" "/public/home/tangwsh/PhysX/build_dcu/CMakeFiles/PhysXSolverGpu.dir/public/home/tangwsh/PhysX/physx/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "PX_DCU_PORT"
  "PX_PHYSX_GPU_STATIC"
  "PX_SUPPORT_GPU_PHYSX"
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
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
