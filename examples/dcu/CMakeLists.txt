cmake_minimum_required(VERSION 3.21)
project(PhysXDcuExamples LANGUAGES CXX)

if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../physx" ABSOLUTE)
endif()

set(PHYSX_INCLUDE_DIRS
    ${PHYSX_ROOT_DIR}/include
    ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/cudamanager/include
    ${PHYSX_ROOT_DIR}/source/common/include
    ${PHYSX_ROOT_DIR}/source/foundation/include
)

find_library(PHYSX_LIB NAMES PhysX PATHS
    ${PHYSX_ROOT_DIR}/../build_cpu
    ${PHYSX_ROOT_DIR}/lib
    /public/home/tangwsh/PhysX/build_cpu
    REQUIRED)

file(GLOB EXAMPLE_SOURCES ${CMAKE_SOURCE_DIR}/*.cpp)

foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${SOURCE_FILE})
    target_include_directories(${EXE_NAME} PRIVATE ${PHYSX_INCLUDE_DIRS})
    target_link_libraries(${EXE_NAME} PRIVATE ${PHYSX_LIB} pthread dl)
endforeach()
