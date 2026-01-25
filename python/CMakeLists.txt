cmake_minimum_required(VERSION 3.21)

project(dolfinx_nanobind LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
  # Windows requires all symbols to be manually exported. This flag exports all
  # symbols automatically, as in Unix.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

find_package(
  Python
  COMPONENTS Interpreter Development ${SKBUILD_SABI_COMPONENT}
  REQUIRED
)

# Options
include(FeatureSummary)

option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(ENABLE_CLANG_TIDY ENABLE_CLANG_TIDY "Run clang-tidy while building")

feature_summary(WHAT ALL)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Enabling C here is a workaround for an ADIOS2 bug, see
# https://github.com/ornladios/ADIOS2/pull/4255. Remove when the ADIOS2
# fix is released.
enable_language(C)

find_package(DOLFINX REQUIRED CONFIG)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

# Create the binding library nanobind handles its own calls to
# target_link_libraries

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_SABI "STABLE_ABI")
endif()

nanobind_add_module(
  cpp
  NB_SUPPRESS_WARNINGS
  NOMINSIZE
  ${NANOBIND_SABI}
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

if(NANOBIND_SABI)
  # mpi4py and petsc4py use generated Cython headers so tell
  # them that the limited API is in use.
  target_compile_definitions(cpp PRIVATE MPI4PY_LIMITED_API=1)
  # Needing to set CYTHON_COMPILING_IN_LIMITED_API is a bug in 
  # Cython-generated API headers, should become unnecessary with
  # mpi4py/petsc4py releases published using Cython >= 3.1.4
  # https://github.com/cython/cython/issues/7108
  target_compile_definitions(cpp PRIVATE CYTHON_COMPILING_IN_LIMITED_API=1)
endif()

# Check for some compiler flags
include(CheckCXXCompilerFlag)

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  list(APPEND DOLFINX_PY_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)
if(HAVE_DEBUG)
  list(APPEND DOLFINX_PY_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)
if(HAVE_O2_OPTIMISATION)
  list(APPEND DOLFINX_PY_CXX_DEVELOPER_FLAGS -O2)
endif()

# Enable C++ standard library debugging
include(CheckCXXSymbolExists)

check_cxx_symbol_exists(_LIBCPP_VERSION "version" LIBCPP)
check_cxx_symbol_exists(__GLIBCXX__ "version" GLIBCXX)

if(LIBCPP)
  list(APPEND DOLFINX_PY_CXX_DEVELOPER_DEFINITIONS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
endif()

if(GLIBCXX)
  list(APPEND DOLFINX_PY_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

# Turn off some checks in gcc12 and gcc13 due to false positives with
# the fmt library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
)
  list(APPEND DOLFINX_PY_CXX_DEVELOPER_FLAGS -Wno-array-bounds;-Wno-stringop-overflow)
endif()

target_compile_options(
  cpp PRIVATE $<$<CONFIG:Developer>:${DOLFINX_PY_CXX_DEVELOPER_FLAGS}>
)
target_compile_definitions(
  cpp PRIVATE $<$<CONFIG:Developer>:${DOLFINX_PY_CXX_DEVELOPER_DEFINITIONS}>
)

target_compile_definitions(cpp PRIVATE cxx_std_20)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(cpp PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../.clang-tidy")
endif()

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
