cmake_minimum_required(VERSION 3.10)

PROJECT(dolfin_pybind11)

set(PYBIND11_CPP_STANDARD -std=c++17)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFIN REQUIRED)

# Strict compiler flags
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
#if (HAVE_PEDANTIC)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic")
#endif()

# Create the binding library
pybind11_add_module(cpp SHARED
  dolfin/wrappers/dolfin.cpp
  dolfin/wrappers/common.cpp
  dolfin/wrappers/fem.cpp
  dolfin/wrappers/function.cpp
  dolfin/wrappers/generation.cpp
  dolfin/wrappers/geometry.cpp
  dolfin/wrappers/graph.cpp
  dolfin/wrappers/io.cpp
  dolfin/wrappers/la.cpp
  dolfin/wrappers/log.cpp
  dolfin/wrappers/mesh.cpp
  dolfin/wrappers/nls.cpp
  dolfin/wrappers/refinement.cpp)

# Add DOLFIN libraries and other config
target_link_libraries(cpp PRIVATE pybind11::module dolfin)

# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Check for petsc4py
find_package(PETSc4py REQUIRED)
if (PETSC4PY_FOUND)
  target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIRS})
endif()

# Check for mpi4py
find_package(MPI4PY REQUIRED)
if (MPI4PY_FOUND)
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
  target_compile_definitions(cpp PRIVATE HAS_PYBIND11_MPI4PY)
endif()
