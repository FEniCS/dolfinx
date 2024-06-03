# ------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFINx
cmake_minimum_required(VERSION 3.21)

# ------------------------------------------------------------------------------
# Set project name and version number
project(DOLFINX VERSION "0.9.0.0")

set(DOXYGEN_DOLFINX_VERSION
    ${DOLFINX_VERSION}
    CACHE STRING "Version for Doxygen" FORCE
)

# ------------------------------------------------------------------------------
# Use C++20
set(CMAKE_CXX_STANDARD 20)

# Require C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
  # Windows requires all symbols to be manually exported. This flag exports all
  # symbols automatically, as in Unix.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# ------------------------------------------------------------------------------
# Get GIT changeset, if available
find_program(GIT_FOUND git)

if(GIT_FOUND)
  # Get the commit hash of the working branch
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(GIT_COMMIT_HASH "unknown")
endif()

# ------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
set(CMAKE_MODULE_PATH "${DOLFINX_SOURCE_DIR}/cmake/modules")

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx)

# ------------------------------------------------------------------------------
# Configurable options for how we want to build
include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINx with shared libraries." ON)
add_feature_info(
  BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINx with shared libraries."
)

# If libdolfinx links to a symbol contained in an external dll, that dll will be
# installed alongside libdolfinx. This excludes system dlls included with
# Windows.
option(INSTALL_RUNTIME_DEPENDENCIES
       "Include runtime dependencies in install (Windows-only)" OFF
)
add_feature_info(
  INSTALL_RUNTIME_DEPENDENCIES INSTALL_RUNTIME_DEPENDENCIES
  "Include runtime dependencies in install (Windows-only)"
)

option(DOLFINX_SKIP_BUILD_TESTS
       "Skip build tests for testing usability of dependency packages." OFF
)
add_feature_info(
  DOLFINX_SKIP_BUILD_TESTS DOLFINX_SKIP_BUILD_TESTS
  "Skip build tests for testing usability of dependency packages."
)

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
       "Add paths to linker search and installed rpath." ON
)
add_feature_info(
  CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
  "Add paths to linker search and installed rpath."
)

# Control Basix discovery
option(
  DOLFINX_BASIX_PYTHON
  "Ask Python basix module for hint where to find Basix C++ install using CONFIG mode."
  ON
)
add_feature_info(
  DOLFINX_BASIX_PYTHON
  DOLFINX_BASIX_PYTHON
  "Ask Python basix module for hint where to find Basix C++ install using CONFIG mode."
)

# Control UFCx discovery
option(
  DOLFINX_UFCX_PYTHON
  "Ask Python FFCx module where to find ufcx.h header using MODULE mode. Otherwise use CONFIG mode."
  ON
)
add_feature_info(
  DOLFINX_UFCX_PYTHON
  DOLFINX_UFCX_PYTHON
  "Ask Python FFCx module where to find ufcx.h header using MODULE mode. Otherwise use CONFIG mode."
)

# ------------------------------------------------------------------------------
# Enable or disable optional packages

if(DOLFINX_ENABLE_ADIOS2)
  set(_REQUIRE_ADIOS2
      TRUE
      CACHE BOOL "Is ADIOS2 REQUIRED?"
  )
else()
  set(_REQUIRE_ADIOS2
      FALSE
      CACHE BOOL "Is ADIOS2 REQUIRED?"
  )
endif()
option(DOLFINX_ENABLE_ADIOS2 "Compile with support for ADIOS2." ON)
set_package_properties(
  ADIOS2 PROPERTIES
  TYPE OPTIONAL
  DESCRIPTION "Adaptable Input/Output (I/O) System."
  URL "https://adios2.readthedocs.io/en/latest/"
  PURPOSE "IO, including in parallel"
)

if(DOLFINX_ENABLE_PETSC)
  set(_REQUIRE_PETSC
      TRUE
      CACHE BOOL "Is PETSc REQUIRED?"
  )
else()
  set(_REQUIRE_PETSC
      FALSE
      CACHE BOOL "Is PETSc REQUIRED?"
  )
endif()

option(DOLFINX_ENABLE_PETSC "Compile with support for PETSc." ON)
set_package_properties(
  PETSc PROPERTIES
  TYPE RECOMMENDED
  DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation"
  URL "https://petsc.org/"
  PURPOSE "Linear and nonlinear solvers"
)

if(DOLFINX_ENABLE_PARMETIS)
  set(_REQUIRE_PARMETIS
      TRUE
      CACHE BOOL "Is Parmetis REQUIRED?"
  )
else()
  set(_REQUIRE_PARMETIS
      FALSE
      CACHE BOOL "Is Parmetis REQUIRED?"
  )
endif()
option(DOLFINX_ENABLE_PARMETIS "Compile with support for ParMETIS." ON)
set_package_properties(
  ParMETIS PROPERTIES
  TYPE RECOMMENDED
  DESCRIPTION "Parallel Graph Partitioning and Fill-reducing Matrix Ordering"
  URL "http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview"
  PURPOSE "Parallel graph partitioning"
)

if(DOLFINX_ENABLE_SCOTCH)
  set(_REQUIRE_SCOTCH
      TRUE
      CACHE BOOL "Is SCOTCH REQUIRED?"
  )
else()
  set(_REQUIRE_SCOTCH
      FALSE
      CACHE BOOL "Is SCOTCH REQUIRED?"
  )
endif()
option(DOLFINX_ENABLE_SCOTCH "Compile with support for SCOTCH." ON)
set_package_properties(
  SCOTCH PROPERTIES
  TYPE OPTIONAL
  DESCRIPTION
    "Programs and libraries for graph, mesh and hypergraph partitioning"
  URL "https://www.labri.fr/perso/pelegrin/scotch"
  PURPOSE "Parallel graph partitioning"
)

if(DOLFINX_ENABLE_SLEPC)
  set(_REQUIRE_SLEPC
      TRUE
      CACHE BOOL "Is SLEPc REQUIRED?"
  )
else()
  set(_REQUIRE_SLEPC
      FALSE
      CACHE BOOL "Is SLEPc REQUIRED?"
  )
endif()
option(DOLFINX_ENABLE_SLEPC "Compile with support for SLEPc." ON)
set_package_properties(
  SLEPc PROPERTIES
  TYPE RECOMMENDED
  DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
  URL "http://slepc.upv.es/"
  PURPOSE "Eigenvalue computation"
)

if(DOLFINX_ENABLE_KAHIP)
  set(_REQUIRE_KAHIP
      TRUE
      CACHE BOOL "Is KaHIP REQUIRED?"
  )
else()
  set(_REQUIRE_KAHIP
      FALSE
      CACHE BOOL "Is KaHIP REQUIRED?"
  )
endif()
option(DOLFINX_ENABLE_KAHIP "Compile with support for KaHIP." OFF)
set_package_properties(
  KaHIP PROPERTIES
  TYPE OPTIONAL
  DESCRIPTION "A family of graph partitioning programs"
  URL "https://kahip.github.io/"
  PURPOSE "Parallel graph partitioning"
)

# ------------------------------------------------------------------------------
# Check for MPI
find_package(MPI 3 REQUIRED)

find_package(spdlog REQUIRED)
# ------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Developer MinSizeRel Release RelWithDebInfo."
        FORCE
  )
endif()

# Check for some compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-pipe HAVE_PIPE)

if(HAVE_PIPE)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -pipe)
endif()

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)

if(HAVE_DEBUG)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)

if(HAVE_O2_OPTIMISATION)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -O2)
endif()

# Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
# library
if(CMAKE_COMPILER_IS_GNUCXX
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS
       -Wno-array-bounds;-Wno-stringop-overflow
  )
endif()

# ------------------------------------------------------------------------------
# Find required packages

# pugixml
find_package(pugixml REQUIRED)

# Note: When updating Boost version, also update DOLFINXConfig.cmake.in
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()

set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED timer)
set_package_properties(
  Boost PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "Boost C++ libraries"
  URL "http://www.boost.org"
)

# Basix C++ files can be installed as a standalone C++ library, or in the Basix
# Python module tree.

# If requested (default), ask the Python interpreter for hints on where to find
# Basix C++ library.
if(DOLFINX_BASIX_PYTHON)
  find_package(
    Python3
    COMPONENTS Interpreter
    QUIET
  )

  if(Python3_Interpreter_FOUND)
    message(STATUS "Checking for Basix hints with ${Python3_EXECUTABLE}")
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE} -c
        "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
      OUTPUT_VARIABLE BASIX_PY_DIR
      RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  if(BASIX_PY_DIR)
    # TODO: CMake 3.20 has more modern cmake_path.
    file(TO_CMAKE_PATH "${BASIX_PY_DIR}" BASIX_PY_DIR)
    message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")

    # Basix installed from manylinux wheel requires rpath set.
    if(IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
      set(CMAKE_INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
    endif()
  else()
    message(STATUS "No Basix hint was found.")
  endif()
endif()

find_package(Basix 0.8 REQUIRED CONFIG HINTS ${BASIX_PY_DIR})
set_package_properties(
  basix PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "FEniCS tabulation library"
  URL "https://github.com/fenics/basix"
)

# Check for HDF5
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_FIND_DEBUG TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)

if(NOT HDF5_IS_PARALLEL)
  message(
    FATAL_ERROR
      "Found serial HDF5 build, MPI HDF5 build required, try setting HDF5_DIR or HDF5_ROOT"
  )
endif()

set_package_properties(
  HDF5 PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
  URL "https://www.hdfgroup.org/HDF5"
)

# Check for UFC Note: we use the case (ufcx vs UFCx) elsewhere to determine by
# which method UFCx was found.
if(NOT DOLFINX_UFCX_PYTHON)
  # Check in CONFIG mode, i.e. look for installed ufcxConfig.cmake
  find_package(ufcx 0.9 REQUIRED CONFIG)
else()
  # Check in MODULE mode (using FindUFCX.cmake) using Python intepreter.
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )
  find_package(UFCx 0.9 REQUIRED MODULE)
endif()

set_package_properties(
  UFCx PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "Interface for form-compilers (part of FFCx)"
  URL "https://github.com/fenics/ffcx"
)

# ------------------------------------------------------------------------------
# Find optional packages
if(DOLFINX_ENABLE_ADIOS2 AND _REQUIRE_ADIOS2)
  find_package(ADIOS2 2.8.1 REQUIRED)
elseif(DOLFINX_ENABLE_ADIOS2)
  find_package(ADIOS2 2.8.1)
endif()
if(ADIOS2_FOUND AND NOT ADIOS2_HAVE_MPI)
  message(
    FATAL_ERROR
      "Found serial ADIOS2 build, MPI ADIOS2 build required, try setting ADIOS2_DIR or ADIOS2_ROOT"
  )
endif()

if(DOLFINX_ENABLE_PETSC)
  find_package(PkgConfig REQUIRED)
  set(ENV{PKG_CONFIG_PATH}
      "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
  )
  if(_REQUIRE_PETSC)
    pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)
  else()
    pkg_search_module(PETSC OPTIONAL IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)
  endif()

  # Setting for FeatureSummary
  if(PETSC_FOUND)
    message(
      STATUS "Found PETSc version ${PETSC_VERSION}, prefix: ${PETSC_PREFIX}"
    )
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND PETSc)
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND PETSc)
  endif()
endif()

if(DOLFINX_ENABLE_SLEPC AND PETSC_FOUND)
  find_package(PkgConfig REQUIRED)
  set(ENV{PKG_CONFIG_PATH}
      "$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{SLEPC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
  )
  set(ENV{PKG_CONFIG_PATH}
      "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
  )
  set(ENV{PKG_CONFIG_PATH}
      "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}:$ENV{PETSC_DIR}:$ENV{PKG_CONFIG_PATH}"
  )
  if(_REQUIRE_SLEPC)
    pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET slepc>=3.15)
  else()
    pkg_search_module(SLEPC IMPORTED_TARGET slepc>=3.15)
  endif()

  # Setting for FeatureSummary
  if(SLEPC_FOUND)
    message(
      STATUS "Found SLEPc version ${SLEPC_VERSION}, prefix: ${SLEPC_PREFIX}"
    )
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND SLEPc)
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND SLEPc)
  endif()
elseif(_REQUIRE_SLEPC AND NOT PETSC_FOUND)
  message(
    FATAL_ERROR
      "SLEPc requested, but no configured because PETSc was not found."
  )
endif()

if(DOLFINX_ENABLE_SCOTCH)
  find_package(SCOTCH CONFIG) # Attempt to find in CONFIG mode
  if(NOT SCOTCH_FOUND)
    if(_REQUIRE_SCOTCH) # If not found in CONFIG mode, try MODULE mode
      find_package(SCOTCH REQUIRED)
    else()
      find_package(SCOTCH)
    endif()
  endif()

  if(TARGET SCOTCH::scotch AND NOT TARGET SCOTCH::ptscotch)
    message(
      STATUS "SCOTCH found, but not PT-SCOTCH (parallel). Not enabling SCOTCH."
    )
    set(SCOTCH_FOUND FALSE)
  endif()
endif()

if(DOLFINX_ENABLE_PARMETIS AND _REQUIRE_PARMETIS)
  find_package(ParMETIS 4.0.2 REQUIRED)
elseif(DOLFINX_ENABLE_PARMETIS)
  find_package(ParMETIS 4.0.2)
endif()

if(DOLFINX_ENABLE_KAHIP AND _REQUIRE_KAHIP)
  find_package(KaHIP REQUIRED)
elseif(DOLFINX_ENABLE_KAHIP)
  find_package(KaHIP)
endif()

# ------------------------------------------------------------------------------
# Print summary of found and not found optional packages
feature_summary(WHAT ALL)

# Check that at least one graph partitioner has been found
if(NOT SCOTCH_FOUND
   AND NOT PARMETIS_FOUND
   AND NOT KAHIP_FOUND
)
  message(
    FATAL_ERROR
      "No graph partitioner found. SCOTCH, ParMETIS or KaHIP is required."
  )
endif()

# ------------------------------------------------------------------------------
# Installation of DOLFINx library
add_subdirectory(dolfinx)

# ------------------------------------------------------------------------------
# Generate and install helper file dolfinx.conf

# FIXME: Can CMake provide the library path name variable?
if(APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

# FIXME: not cross-platform compatible Create and install dolfinx.conf file
configure_file(
  ${DOLFINX_SOURCE_DIR}/cmake/templates/dolfinx.conf.in
  ${CMAKE_BINARY_DIR}/dolfinx.conf @ONLY
)
install(
  FILES ${CMAKE_BINARY_DIR}/dolfinx.conf
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/dolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Install the demo source files
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/demo
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx
  FILES_MATCHING
  PATTERN "CMakeLists.txt"
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.c"
  PATTERN "*.cpp"
  PATTERN "*.py"
  PATTERN "*.xdmf"
  PATTERN "*.h5"
  PATTERN "CMakeFiles" EXCLUDE
)

# ------------------------------------------------------------------------------
# Add "make uninstall" target
configure_file(
  "${DOLFINX_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY
)

add_custom_target(
  uninstall "${CMAKE_COMMAND}" -P
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
)

# ------------------------------------------------------------------------------
# Print post-install message
add_subdirectory(cmake/post-install)

# ------------------------------------------------------------------------------
