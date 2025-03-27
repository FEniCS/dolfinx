cmake_minimum_required(VERSION 3.21)

project(dolfinx_nanobind)

if(WIN32)
  # Windows requires all symbols to be manually exported. This flag exports all
  # symbols automatically, as in Unix.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(DOLFINX REQUIRED CONFIG)

set(CMAKE_MODULE_PATH /Users/chrisrichardson/Work/dolfinx/cpp/cmake/modules)

find_package(GMP)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

# Create the binding library nanobind handles its own calls to
# target_link_libraries
nanobind_add_module(
  cpp
  NB_SUPPRESS_WARNINGS
  NOMINSIZE
  dolfinx/wrappers/dolfinx.cpp
  dolfinx/wrappers/assemble.cpp
  dolfinx/wrappers/common.cpp
  dolfinx/wrappers/fem.cpp
  dolfinx/wrappers/geometry.cpp
  dolfinx/wrappers/graph.cpp
  dolfinx/wrappers/io.cpp
  dolfinx/wrappers/la.cpp
  dolfinx/wrappers/log.cpp
  dolfinx/wrappers/mesh.cpp
  dolfinx/wrappers/petsc.cpp
  dolfinx/wrappers/refinement.cpp
)

# Add strict compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)
if(HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-Wextra;-pedantic)

  # Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
  # library
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
     AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
     AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
  )
    target_compile_options(cpp PRIVATE -Wno-array-bounds;-Wno-stringop-overflow)
  endif()
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_definitions(cpp PRIVATE cxx_std_20)

# Add DOLFINx libraries
target_link_libraries(cpp PRIVATE dolfinx)
# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(cpp PRIVATE bcrypt)
endif()

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
  target_compile_definitions(cpp PRIVATE HAS_PETSC4PY)
else()
  message(STATUS "petsc4py not found.")
endif()

# Check for mpi4py
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cpp DESTINATION dolfinx)
