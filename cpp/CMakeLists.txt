# ------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFINx
cmake_minimum_required(VERSION 3.26)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW) # Boost CONFIG mode
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0144.html
endif()

# ------------------------------------------------------------------------------
# Set project name and version number
project(
  DOLFINX
  VERSION "0.12.0.0"
  DESCRIPTION
    "The finite element problem solving environment of the FEniCS Project"
  HOMEPAGE_URL "https://fenicsproject.org"
  LANGUAGES CXX C
)

set(
  DOXYGEN_DOLFINX_VERSION
  ${DOLFINX_VERSION}
  CACHE STRING
  "Version for Doxygen"
  FORCE
)

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Get GIT changeset, if available
find_package(Git QUIET)

if(GIT_FOUND)
  # Get the commit hash of the working branch
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${DOLFINX_SOURCE_DIR}
    RESULT_VARIABLE GIT_RESULT
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT GIT_RESULT EQUAL 0)
    set(GIT_COMMIT_HASH "unknown")
  endif()
else()
  set(GIT_COMMIT_HASH "unknown")
endif()

# ------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
list(APPEND CMAKE_MODULE_PATH "${DOLFINX_SOURCE_DIR}/cmake/modules")

# PETSc/SLEPc pkg-config discovery helpers, shared with DOLFINXConfig.cmake
include(DolfinxPkgConfigHelpers)

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/dolfinx")

# ------------------------------------------------------------------------------
# Configurable options for how we want to build
include(FeatureSummary)
include(GNUInstallDirs)

# Declare a build option and register it with FeatureSummary, so that the
# description is written once rather than repeated in add_feature_info.
function(dolfinx_option name description default)
  option(${name} "${description}" ${default})
  add_feature_info(${name} ${name} "${description}")
endfunction()

dolfinx_option(BUILD_SHARED_LIBS "Build DOLFINx with shared libraries." ON)

# If libdolfinx links to a symbol contained in an external dll, that dll will be
# installed alongside libdolfinx. This excludes system dlls included with
# Windows.
dolfinx_option(
  INSTALL_RUNTIME_DEPENDENCIES
  "Include runtime dependencies in install (Windows-only)"
  OFF
)

dolfinx_option(
  DOLFINX_SKIP_BUILD_TESTS
  "Skip build tests for testing usability of dependency packages."
  OFF
)

# Add all shared library paths to rpath so libs in non-system paths are found.
# This is on by default because many users build against non-system paths.
# Package builders may want to set this to FALSE.
dolfinx_option(
  DOLFINX_INSTALL_RPATH_USE_LINK_PATH
  "Add linker search paths to rpath of dolfinx target. Packagers may want to set to FALSE."
  TRUE
)

# Control Basix discovery
dolfinx_option(
  DOLFINX_BASIX_PYTHON
  "Ask Python basix module for hint where to find Basix C++ install using CONFIG mode."
  ON
)

# Control UFCx discovery
dolfinx_option(
  DOLFINX_UFCX_PYTHON
  "Ask Python FFCx module where to find ufcx.h header using MODULE mode. Otherwise use CONFIG mode."
  ON
)

# Control PETSc discovery
dolfinx_option(
  DOLFINX_PETSC_PYTHON
  "Ask Python petsc module for a hint where to find PETSc."
  ON
)

# Control SLEPc discovery
dolfinx_option(
  DOLFINX_SLEPC_PYTHON
  "Ask Python slepc module for a hint where to find SLEPc."
  ON
)

# ------------------------------------------------------------------------------
# Enable or disable optional packages

# Declare an optional-dependency switch DOLFINX_ENABLE_<X>.
#
# Each switch is AUTO (the default), ON (required) or OFF (disabled). AUTO
# discovers a package without requiring it, while ON makes a missing package a
# configuration error. This avoids inferring the user's intent from the cache
# entry type, which cannot distinguish a default from an explicit ON after the
# first configure.
#
# Existing BOOL cache entries are migrated: a previous explicit requirement is
# retained, otherwise an enabled entry becomes AUTO. This preserves the old
# optional default while allowing a subsequent -DDOLFINX_ENABLE_<X>=ON to make
# the package required.
#
# Sets, for use at the find_package()/pkg_search_module() call site:
#   _DOLFINX_USE_<X>      TRUE/FALSE
#   _REQUIRE_<X>          TRUE/FALSE
#   _REQUIRE_<X>_KEYWORD  "REQUIRED" or "" (empty)
function(dolfinx_enable_option name description)
  string(REPLACE "DOLFINX_ENABLE_" "_REQUIRE_" require_var "${name}")
  string(REPLACE "DOLFINX_ENABLE_" "_DOLFINX_USE_" use_var "${name}")
  get_property(cache_type CACHE ${name} PROPERTY TYPE)

  # Prior versions used a BOOL switch. A default ON meant AUTO, while an
  # explicit ON was recorded in the internal requirement cache.
  if(cache_type STREQUAL "BOOL")
    if(${name})
      if(DEFINED ${require_var} AND ${require_var})
        set(mode ON)
      else()
        set(mode AUTO)
      endif()
    else()
      set(mode OFF)
    endif()
    set(${name} ${mode} CACHE STRING "${description}" FORCE)
  else()
    set(${name} AUTO CACHE STRING "${description}")
    string(TOUPPER "${${name}}" mode)
  endif()

  set_property(CACHE ${name} PROPERTY STRINGS AUTO ON OFF)
  if(mode STREQUAL "AUTO")
    set(${use_var} TRUE)
    set(${use_var} TRUE PARENT_SCOPE)
    set(${require_var} FALSE PARENT_SCOPE)
    set(${require_var}_KEYWORD "" PARENT_SCOPE)
  elseif(mode MATCHES "^(ON|TRUE|YES|Y|1)$")
    set(${name} ON CACHE STRING "${description}" FORCE)
    set(${use_var} TRUE)
    set(${use_var} TRUE PARENT_SCOPE)
    set(${require_var} TRUE PARENT_SCOPE)
    set(${require_var}_KEYWORD "REQUIRED" PARENT_SCOPE)
  elseif(
    mode MATCHES "^(OFF|FALSE|NO|N|IGNORE|NOTFOUND|0)$"
    OR mode MATCHES "-NOTFOUND$"
  )
    set(${name} OFF CACHE STRING "${description}" FORCE)
    set(${use_var} FALSE)
    set(${use_var} FALSE PARENT_SCOPE)
    set(${require_var} FALSE PARENT_SCOPE)
    set(${require_var}_KEYWORD "" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "${name} must be AUTO, ON or OFF, not '${${name}}'.")
  endif()
  add_feature_info(${name} ${use_var} "${description}")
endfunction()

dolfinx_enable_option(DOLFINX_ENABLE_ADIOS2 "Compile with support for ADIOS2.")
set_package_properties(
  ADIOS2
  PROPERTIES
    TYPE OPTIONAL
    DESCRIPTION "Adaptable Input/Output (I/O) System."
    URL "https://adios2.readthedocs.io/en/latest/"
    PURPOSE "IO, including in parallel"
)

dolfinx_enable_option(DOLFINX_ENABLE_PETSC "Compile with support for PETSc.")
set_package_properties(
  PETSc
  PROPERTIES
    TYPE RECOMMENDED
    DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation"
    URL "https://petsc.org/"
    PURPOSE "Linear and nonlinear solvers"
)

dolfinx_enable_option(
  DOLFINX_ENABLE_PARMETIS
  "Compile with support for ParMETIS."
)
set_package_properties(
  ParMETIS
  PROPERTIES
    TYPE RECOMMENDED
    DESCRIPTION "Parallel Graph Partitioning and Fill-reducing Matrix Ordering"
    URL "http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview"
    PURPOSE "Parallel graph partitioning"
)

dolfinx_enable_option(DOLFINX_ENABLE_SCOTCH "Compile with support for SCOTCH.")
set_package_properties(
  SCOTCH
  PROPERTIES
    TYPE OPTIONAL
    DESCRIPTION
      "Programs and libraries for graph, mesh and hypergraph partitioning"
    URL "https://www.labri.fr/perso/pelegrin/scotch"
    PURPOSE "Parallel graph partitioning"
)

dolfinx_enable_option(DOLFINX_ENABLE_SLEPC "Compile with support for SLEPc.")
set_package_properties(
  SLEPc
  PROPERTIES
    TYPE RECOMMENDED
    DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
    URL "http://slepc.upv.es/"
    PURPOSE "Eigenvalue computation"
)

dolfinx_enable_option(DOLFINX_ENABLE_KAHIP "Compile with support for KaHIP.")
set_package_properties(
  KaHIP
  PROPERTIES
    TYPE OPTIONAL
    DESCRIPTION "A family of graph partitioning programs"
    URL "https://kahip.github.io/"
    PURPOSE "Parallel graph partitioning"
)

dolfinx_enable_option(
  DOLFINX_ENABLE_SUPERLU_DIST
  "Compile with support for SuperLU_DIST."
)
set_package_properties(
  SuperLU_DIST
  PROPERTIES
    TYPE OPTIONAL
    DESCRIPTION "Distributed direct LU Solver"
    URL "https://portal.nersc.gov/project/sparse/superlu/"
    PURPOSE "Linear Algebra Solver"
)

# ------------------------------------------------------------------------------
# Check for MPI
find_package(MPI 3 REQUIRED COMPONENTS CXX)

# ------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user). Multi-config generators
# pick the configuration at build time and ignore CMAKE_BUILD_TYPE.
set(
  DOLFINX_BUILD_TYPES
  Debug
  Developer
  MinSizeRel
  Release
  RelWithDebInfo
)
if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
    set(
      CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING
      "Choose the type of build, options are: ${DOLFINX_BUILD_TYPES}."
      FORCE
    )
  endif()
  # Only a cache entry can carry STRINGS. On a multi-config generator no
  # entry is created above, and a parent project may have set
  # CMAKE_BUILD_TYPE as an ordinary variable.
  if(DEFINED CACHE{CMAKE_BUILD_TYPE})
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${DOLFINX_BUILD_TYPES})
  endif()
endif()

include(DolfinxDeveloperCompilerFlags)

# ------------------------------------------------------------------------------
# Find required packages

find_package(pugixml REQUIRED)
find_package(spdlog REQUIRED)

# Note: When updating Boost version, also update DOLFINXConfig.cmake.in
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()

if(DEFINED ENV{BOOST_USE_MULTITHREADED})
  set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
endif()
# 1.81 is required for boost/unordered/unordered_flat_set.hpp, used by
# dolfinx::graph::partitioners.
find_package(Boost 1.81 REQUIRED)
set_package_properties(
  Boost
  PROPERTIES
    TYPE REQUIRED
    DESCRIPTION "Boost C++ libraries"
    URL "http://www.boost.org"
)

# Basix C++ files can be installed as a standalone C++ library, or in
# the Basix Python module tree.

# If requested (default), ask the Python interpreter for hints on where
# to find Basix C++ library.
if(DOLFINX_BASIX_PYTHON)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    message(STATUS "Checking for Basix hints with ${Python3_EXECUTABLE}")
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE} -c
        "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
      OUTPUT_VARIABLE BASIX_PY_DIR
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  if(BASIX_PY_DIR)
    # Converts os native to cmake native path
    cmake_path(SET BASIX_PY_DIR "${BASIX_PY_DIR}")
    cmake_path(GET BASIX_PY_DIR PARENT_PATH BASIX_PY_PARENT_DIR)

    # These are nullops if the directories don't exist/contain nothing.
    # Basix lib may be in . (e.g. manual combined pip install)
    list(APPEND CMAKE_INSTALL_RPATH "${BASIX_PY_DIR}")
    # Basix lib installed from cibuildwheel wheel may be in ../fenics_basix.libs
    list(APPEND CMAKE_INSTALL_RPATH "${BASIX_PY_PARENT_DIR}/fenics_basix.libs")

    message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
    list(APPEND CMAKE_PREFIX_PATH ${BASIX_PY_DIR})
  else()
    message(STATUS "No Basix hint was found.")
  endif()
endif()

find_package(Basix 0.12 REQUIRED CONFIG)
set_package_properties(
  basix
  PROPERTIES
    TYPE REQUIRED
    DESCRIPTION "FEniCS tabulation library"
    URL "https://github.com/fenics/basix"
)

# Check for HDF5
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
# HDF5_IS_PARALLEL -> HDF5_PROVIDES_PARALLEL in HDF5@2.0.0
if(NOT (HDF5_IS_PARALLEL OR HDF5_PROVIDES_PARALLEL))
  message(
    FATAL_ERROR
    "Found serial HDF5, MPI HDF5 build required. Try setting HDF5_DIR or HDF5_ROOT."
  )
endif()

set_package_properties(
  HDF5
  PROPERTIES
    TYPE REQUIRED
    DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
    URL "https://www.hdfgroup.org/HDF5"
)

# Check for UFC
# Note: we use the case (ufcx vs UFCx) elsewhere to determine by which
# method UFCx was found.
if(NOT DOLFINX_UFCX_PYTHON)
  # Check in CONFIG mode, i.e. look for installed ufcxConfig.cmake
  find_package(ufcx 0.12 REQUIRED CONFIG)
else()
  # Check in MODULE mode (using FindUFCX.cmake) using Python
  # interpreter
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  find_package(UFCx 0.12 REQUIRED MODULE)
endif()

set_package_properties(
  UFCx
  PROPERTIES
    TYPE REQUIRED
    DESCRIPTION "Interface for form-compilers (part of FFCx)"
    URL "https://github.com/fenics/ffcx"
)

# ------------------------------------------------------------------------------
# Find optional packages
if(_DOLFINX_USE_ADIOS2)
  find_package(ADIOS2 2.8.1 ${_REQUIRE_ADIOS2_KEYWORD} COMPONENTS CXX MPI)
endif()

if(_DOLFINX_USE_PETSC)
  find_package(PkgConfig REQUIRED)
  if(DOLFINX_PETSC_PYTHON)
    dolfinx_python_prefix_hint(petsc PETSC_DIR)
  endif()
  dolfinx_prepend_pkgconfig_path(PETSC_DIR)

  pkg_search_module(
    PETSC
    ${_REQUIRE_PETSC_KEYWORD}
    IMPORTED_TARGET
    PETSc>=3.15
    petsc>=3.15
  )

  # Setting for FeatureSummary
  if(PETSC_FOUND)
    message(
      STATUS
      "Found PETSc version ${PETSC_VERSION}, prefix: ${PETSC_PREFIX}"
    )
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND PETSc)
    list(APPEND CMAKE_PREFIX_PATH ${PETSC_PREFIX})
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND PETSc)
  endif()
endif()

if(_DOLFINX_USE_SLEPC AND TARGET PkgConfig::PETSC)
  find_package(PkgConfig REQUIRED)
  if(DOLFINX_SLEPC_PYTHON)
    dolfinx_python_prefix_hint(slepc SLEPC_DIR)
  endif()
  dolfinx_prepend_pkgconfig_path(SLEPC_DIR)

  pkg_search_module(SLEPC ${_REQUIRE_SLEPC_KEYWORD} IMPORTED_TARGET slepc>=3.15)

  # Setting for FeatureSummary
  if(SLEPC_FOUND)
    message(
      STATUS
      "Found SLEPc version ${SLEPC_VERSION}, prefix: ${SLEPC_PREFIX}"
    )
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND SLEPc)
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND SLEPc)
  endif()
elseif(_REQUIRE_SLEPC AND NOT TARGET PkgConfig::PETSC)
  message(
    FATAL_ERROR
    "SLEPc required, but not configured because PETSc was not found."
  )
endif()

# SCOTCH 7.0.1 was first version with ScotchConfig.cmake
if(_DOLFINX_USE_SCOTCH)
  find_package(SCOTCH 7.0.1 ${_REQUIRE_SCOTCH_KEYWORD} CONFIG)
endif()

if(TARGET SCOTCH::scotch AND NOT TARGET SCOTCH::ptscotch)
  message(
    STATUS
    "SCOTCH found, but not PT-SCOTCH (parallel). Not enabling SCOTCH."
  )
  set(SCOTCH_FOUND FALSE)
  if(_REQUIRE_SCOTCH)
    message(FATAL_ERROR "SCOTCH requested, but not found.")
  endif()
endif()

if(SCOTCH_FOUND)
  message(
    STATUS
    "Found PT-SCOTCH version ${SCOTCH_VERSION}, include dir: ${SCOTCH_INCLUDE_DIR}, lib dir: ${SCOTCH_LIBRARY_DIR}"
  )
endif()

if(_DOLFINX_USE_PARMETIS)
  find_package(ParMETIS 4.0.2 ${_REQUIRE_PARMETIS_KEYWORD})
endif()

if(_DOLFINX_USE_KAHIP)
  find_package(KaHIP ${_REQUIRE_KAHIP_KEYWORD})
endif()

if(_DOLFINX_USE_SUPERLU_DIST)
  find_package(PkgConfig REQUIRED)
  # If PETSc was built with SuperLU_DIST then pickup install there
  dolfinx_prepend_pkgconfig_path(PETSC_DIR)

  pkg_search_module(
    SUPERLU_DIST
    ${_REQUIRE_SUPERLU_DIST_KEYWORD}
    IMPORTED_TARGET
    superlu_dist
  )

  if(SUPERLU_DIST_FOUND)
    message(
      STATUS
      "Found SuperLU_DIST version ${SUPERLU_DIST_VERSION}, prefix: ${SUPERLU_DIST_PREFIX}"
    )
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND SuperLU_DIST)
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND SuperLU_DIST)
  endif()
endif()

# ------------------------------------------------------------------------------
# Print summary of found and not found optional packages
feature_summary(WHAT ALL)

# Check that at least one graph partitioner has been found
if(
  NOT TARGET SCOTCH::ptscotch
  AND NOT TARGET ParMETIS::ParMETIS
  AND NOT TARGET KaHIP::KaHIP
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
# Unit testing
option(BUILD_TESTING "Build DOLFINx unit tests and create test target." OFF)
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# ------------------------------------------------------------------------------
# Demos
option(
  DOLFINX_BUILD_DEMOS
  "Build DOLFINx C++ demos and create demo test targets."
  OFF
)
if(DOLFINX_BUILD_DEMOS)
  add_subdirectory(demo)
endif()

# ------------------------------------------------------------------------------
# Generate and install helper file dolfinx.conf

# FIXME: Can CMake provide the library path name variable?
if(APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

# FIXME: not cross-platform compatible
# Create and install dolfinx.conf file
configure_file(
  ${DOLFINX_SOURCE_DIR}/cmake/templates/dolfinx.conf.in
  ${PROJECT_BINARY_DIR}/dolfinx.conf
  @ONLY
)
install(
  FILES ${PROJECT_BINARY_DIR}/dolfinx.conf
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/dolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Install the demo source files
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/demo
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx
  FILES_MATCHING
  PATTERN "CMakeLists.txt"
  PATTERN "README.md"
  PATTERN "*.cpp"
  PATTERN "*.py"
  # Guard against in-source build detritus, e.g. CMakeCXXCompilerId.cpp
  PATTERN "CMakeFiles" EXCLUDE
)

# ------------------------------------------------------------------------------
# Add "make uninstall" target
configure_file(
  "${DOLFINX_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  @ONLY
)

add_custom_target(
  uninstall
  COMMAND
    ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  COMMENT "Uninstalling files listed in install_manifest.txt"
  VERBATIM
)

# ------------------------------------------------------------------------------
# Print post-install message
#
# The message body is a bracket argument, so its ${...} are expanded when
# the install script runs rather than at configure time. That way the
# reported prefix is the one actually installed to, which
# `cmake --install . --prefix <dir>` overrides.
install(
  CODE
    "set(DOLFINX_INSTALL_LIBDIR \"${CMAKE_INSTALL_LIBDIR}\")
set(DOLFINX_INSTALL_DATADIR \"${CMAKE_INSTALL_DATAROOTDIR}\")"
)
install(
  CODE
    [[
message(
  "----------------------------------------------------------------------------
DOLFINx has now been installed in

    ${CMAKE_INSTALL_PREFIX}

and demo programs have been installed in

    ${CMAKE_INSTALL_PREFIX}/${DOLFINX_INSTALL_DATADIR}/dolfinx/demo

Don't forget to update your environment variables. This can be done
easily using the helper file 'dolfinx.conf' which sets the appropriate
variables (for users of the Bash shell).

To update your environment variables, run the following command:

    source ${CMAKE_INSTALL_PREFIX}/${DOLFINX_INSTALL_LIBDIR}/dolfinx/dolfinx.conf

----------------------------------------------------------------------------")
]]
)

# ------------------------------------------------------------------------------
