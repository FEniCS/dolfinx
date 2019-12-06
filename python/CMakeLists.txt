cmake_minimum_required(VERSION 3.5.0)

PROJECT(dolfin_pybind11)

set(PYBIND11_CPP_STANDARD -std=c++17)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Strict compiler flags
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
#if (HAVE_PEDANTIC)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -pedantic")
#endif()

# Create the binding library
pybind11_add_module(cpp SHARED
  src/dolfin.cpp
  src/common.cpp
  src/fem.cpp
  src/function.cpp
  src/generation.cpp
  src/geometry.cpp
  src/graph.cpp
  src/io.cpp
  src/la.cpp
  src/log.cpp
  src/mesh.cpp
  src/nls.cpp
  src/refinement.cpp)

# Add DOLFIN libraries and other config
target_link_libraries(cpp PRIVATE pybind11::module dolfin)

# Install headers of the binding library
install(FILES src/caster_mpi.h src/caster_petsc.h src/MPICommWrapper.h
  DESTINATION ${DOLFIN_INCLUDE_DIRS}/dolfin/pybind11
  COMPONENT Development)

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
