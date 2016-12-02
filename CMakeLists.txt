# Top level CMakeLists.txt file for DOLFIN

# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFIN)
set(DOLFIN_VERSION_RELEASE 0)
set(DOLFIN_VERSION_MAJOR "2017")
set(DOLFIN_VERSION_MINOR "1")
set(DOLFIN_VERSION_MICRO "0")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")
if (NOT DOLFIN_VERSION_RELEASE)
  set(DOLFIN_VERSION "${DOLFIN_VERSION}.dev0")
endif()

#------------------------------------------------------------------------------
# Require and use C++11

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# Require C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# Check compiler version

# Check for GCC version - earlier versions have insuffcient C++11
# support, or bugs.
if (CMAKE_COMPILER_IS_GNUCXX)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.3)
    message(FATAL_ERROR "GCC version must be at least 4.8.3 (for sufficient C++11  support and to avoid some compiler bugs). You have version ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
endif()

#------------------------------------------------------------------------------
# Get GIT changeset, if available

# Check for git
find_program(GIT_FOUND git)

if (GIT_FOUND)
  # Get the commit hash of the working branch
  execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
  set(GIT_COMMIT_HASH "unknown")
endif()

#------------------------------------------------------------------------------
# General configuration

# Set CMake options, see `cmake --help-policy CMP000x`
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0004 NEW)
  cmake_policy(SET CMP0042 NEW)
endif()

# Set location of our FindFoo.cmake modules
set(DOLFIN_CMAKE_DIR "${DOLFIN_SOURCE_DIR}/cmake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}/modules")

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfin)

#------------------------------------------------------------------------------
# Configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
option(CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects." OFF)
option(DOLFIN_AUTO_DETECT_MPI "Detect MPI automatically (turn this off to use the MPI compiler wrappers directly via setting CXX, CXX, FC)." ON)
option(DOLFIN_DEPRECATION_ERROR "Turn deprecation warnings into errors." OFF)
option(DOLFIN_ENABLE_BENCHMARKS "Enable benchmark programs." OFF)
option(DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(DOLFIN_ENABLE_DOCS "Enable generation of documentation." ON)
option(DOLFIN_ENABLE_GTEST "Enable C++ unit tests with Google Test if DOLFIN_ENABLE_TESTING is true (requires Internet connection to download Google Test when first configured)." ON)
option(DOLFIN_ENABLE_TESTING "Enable testing." OFF)
option(DOLFIN_IGNORE_PETSC4PY_VERSION "Ignore version of PETSc4py." OFF)
option(DOLFIN_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages." OFF)
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)

add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFIN with shared libraries.")
add_feature_info(CMAKE_USE_RELATIVE_PATHS CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects.")
add_feature_info(DOLFIN_AUTO_DETECT_MPI DOLFIN_AUTO_DETECT_MPI "Detect MPI automatically (turn this off to use the MPI compiler wrappers directly via setting CXX, CXX, FC).")
add_feature_info(DOLFIN_ENABLE_CODE_COVERAGE DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage.")
add_feature_info(DOLFIN_WITH_LIBRARY_VERSION DOLFIN_WITH_LIBRARY_VERSION "Build with library version information.")
add_feature_info(DOLFIN_ENABLE_TESTING DOLFIN_ENABLE_TESTING "Enable testing.")
add_feature_info(DOLFIN_ENABLE_GTEST DOLFIN_ENABLE_GTEST "Enable C++ unit tests with Google Test if DOLFIN_ENABLE_TESTING is true (requires Internet connection to download Google Test when first configured).")
add_feature_info(DOLFIN_ENABLE_BENCHMARKS DOLFIN_ENABLE_BENCHMARKS "Enable benchmark programs.")
add_feature_info(DOLFIN_ENABLE_DOCS DOLFIN_ENABLE_DOCS "Enable generation of documentation.")
add_feature_info(DOLFIN_SKIP_BUILD_TESTS DOLFIN_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages.")
add_feature_info(DOLFIN_DEPRECATION_ERROR DOLFIN_DEPRECATION_ERROR "Turn deprecation warnings into errors.")
add_feature_info(DOLFIN_IGNORE_PETSC4PY_VERSION DOLFIN_IGNORE_PETSC4PY_VERSION "Ignore version of PETSc4py.")

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# Hande RPATH on OSX when not installing to a system directory, see
# https://groups.google.com/d/msg/fenics-dev/KSCrob4M_1M/zsJwdN-SCAAJ
# and https://cmake.org/Wiki/CMake_RPATH_handling#Always_full_RPATH.
if (APPLE)
  # The RPATH to be used when installing, but only if it's not a
  # system directory
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
     SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF("${isSystemDir}" STREQUAL "-1")
endif()

#------------------------------------------------------------------------------
# This should be removed when Clang support OpenMP is ready
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  option(DOLFIN_ENABLE_OPENMP "Enable OpenMP." OFF)
else()
  option(DOLFIN_ENABLE_OPENMP "Enable OpenMP." ON)
endif()

#------------------------------------------------------------------------------
# Enable or disable optional packages

# List optional packages
set(OPTIONAL_PACKAGES "")
list(APPEND OPTIONAL_PACKAGES "OpenMP")
list(APPEND OPTIONAL_PACKAGES "MPI")
list(APPEND OPTIONAL_PACKAGES "PETSc")
list(APPEND OPTIONAL_PACKAGES "PETSc4py")
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "SLEPc4py")
list(APPEND OPTIONAL_PACKAGES "Trilinos")
list(APPEND OPTIONAL_PACKAGES "UMFPACK")
list(APPEND OPTIONAL_PACKAGES "CHOLMOD")
list(APPEND OPTIONAL_PACKAGES "SCOTCH")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")
list(APPEND OPTIONAL_PACKAGES "zlib")
list(APPEND OPTIONAL_PACKAGES "Python")
list(APPEND OPTIONAL_PACKAGES "Sphinx")
list(APPEND OPTIONAL_PACKAGES "HDF5")
list(APPEND OPTIONAL_PACKAGES "VTK")

# Add options
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "DOLFIN_ENABLE_${OPTIONAL_PACKAGE}" OPTION_NAME)
  option(${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}." ON)
  add_feature_info(${OPTION_NAME} ${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}.")
endforeach()

# Python version
option(DOLFIN_USE_PYTHON3 "Use Python 3." OFF)

#------------------------------------------------------------------------------
# General compiler defintions

if (DOLFIN_DEPRECATION_ERROR)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DDOLFIN_DEPRECATION_ERROR")
endif()

#------------------------------------------------------------------------------
# Package-specific options

# None at the moment

#------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Developer MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Check for some compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-pipe HAVE_PIPE)
if (HAVE_PIPE)
  set(DOLFIN_CXX_DEVELOPER_FLAGS "-pipe ${DOLFIN_CXX_DEVELOPER_FLAGS}")
endif()

# Add some strict compiler checks
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  set(DOLFIN_CXX_DEVELOPER_FLAGS "-Wall -Werror -pedantic ${DOLFIN_CXX_DEVELOPER_FLAGS}")
endif()

# Debug flags
CHECK_CXX_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  set(DOLFIN_CXX_DEVELOPER_FLAGS "-g ${DOLFIN_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  set(DOLFIN_CXX_DEVELOPER_FLAGS "-O2 ${DOLFIN_CXX_DEVELOPER_FLAGS}")
endif()

# Set 'Developer' build type flags
set(CMAKE_CXX_FLAGS_DEVELOPER "${DOLFIN_CXX_DEVELOPER_FLAGS}" CACHE STRING
  "Flags used by the compiler during development." FORCE)

# FIXME: Do we want to add -DDEBUG to RelWithDebInfo?

# Set debug definitions
if (CMAKE_BUILD_TYPE STREQUAL "Developer" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DDEBUG")
else()
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DNDEBUG")
endif()

# Add flags for generating code coverage reports
if (DOLFIN_ENABLE_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# Settings for Intel compilers
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # Use -isystem incluse flag with Intel compiler
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")

  # Stop spurious warnings from older Intel compilers
  if("${CMAKE_CXX_COMPILER_VERSION}" LESS "13")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd654,1125")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -wd654,1125")
    set(CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER} -wd654,1125")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -wd654,1125")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -wd654,1125")
  endif()
endif()

# Set system include flags to get around CMake bug on OSX with gcc See
# http://public.kitware.com/Bug/print_bug_page.php?bug_id=10837
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif()

if (APPLE)
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
  set(CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER} -Wno-long-long")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-long-long")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wno-long-long")
endif()

#------------------------------------------------------------------------------
# Check type sizes

include(CheckTypeSize)

# Check size of size_t
check_type_size("size_t" size_t_size)
if (size_t_size)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DDOLFIN_SIZE_T=${size_t_size}")
endif()

#------------------------------------------------------------------------------
# Check for MPI and OpenMP

# FIXME: Should be set CMake to use the MPI compiler wrappers?

if (DOLFIN_ENABLE_MPI)
  if (DOLFIN_AUTO_DETECT_MPI)
    find_package(MPI)
    set_package_properties(MPI PROPERTIES TYPE OPTIONAL
      DESCRIPTION "Message Passing Interface (MPI)"
      PURPOSE "Enables DOLFIN to run in parallel with MPI")
  else()
    # Assume user has set MPI compiler wrappers (via CXX, etc or
    # CMAKE_CXX_COMPILER, etc)
    set(MPI_CXX_FOUND TRUE)
    set(MPI_C_FOUND TRUE)
  endif()
endif()

# Test for OpenMP (but not for Clang, which does not support OpenMP)
if (DOLFIN_ENABLE_OPENMP)
  find_package(OpenMP)
  set_package_properties(OpenMP PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Open Multi-Processing API"
    PURPOSE "Enables DOLFIN to run in parallel with OpenMP")
  include(CheckOpenMP)
  check_openmp_unsigned_int_loop_control_variable(OPENMP_UINT_TEST_RUNS)
  if (NOT OPENMP_UINT_TEST_RUNS)
    set(OPENMP_FOUND FALSE)
  endif()
endif()

#------------------------------------------------------------------------------
# Run tests to find required packages

# Check for Boost
set(BOOST_ROOT $ENV{BOOST_DIR} $ENV{BOOST_HOME})
if (BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()

# Prevent FindBoost.cmake from looking for system Boost{foo}.cmake
# files
set(Boost_NO_BOOST_CMAKE true)

set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
find_package(Boost 1.48 QUIET REQUIRED)

# Boost public/private libraries to link to.
# Note: These should all be private as they do not appear in the
# DOLFIN public interface , but there is a linking issues with older
# Boost or CMake. Ubuntu 16.04 requires linking DOLFIN programs with
# filesystem, whereas Ubuntu 16.10 and macOS (Homebrew) do not.
if (Boost_VERSION VERSION_LESS 106100)
  set(DOLFIN_BOOST_COMPONENTS_PUBLIC filesystem)
  set(DOLFIN_BOOST_COMPONENTS_PRIVATE program_options iostreams timer)
else()
  set(DOLFIN_BOOST_COMPONENTS_PUBLIC)
  set(DOLFIN_BOOST_COMPONENTS_PRIVATE filesystem program_options iostreams
    timer)
endif()

# Find required Boost libraries
find_package(Boost COMPONENTS
  ${DOLFIN_BOOST_COMPONENTS_PUBLIC} ${DOLFIN_BOOST_COMPONENTS_PRIVATE} REQUIRED)
set_package_properties(Boost PROPERTIES TYPE REQUIRED
  DESCRIPTION "Boost C++ libraries"
  URL "http://www.boost.org")

# Check for required package Eigen3
find_package(Eigen3 3.2.0 REQUIRED)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Lightweight C++ template library for linear algebra"
  URL "http://eigen.tuxfamily.org")

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Note: Check for Python interpreter even when Python is disabled
#       because it is used to get the installation path for
#       dolfin_utils
if (DOLFIN_USE_PYTHON3)
  find_package(PythonInterp 3)
else()
  find_package(PythonInterp 2.7)
endif()

set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED
  DESCRIPTION "Interactive high-level object-oriented language"
  URL "http://www.python.org")

if (DOLFIN_ENABLE_PYTHON)
  set_package_properties(PythonInterp PROPERTIES PURPOSE "Needed for the Python interface to DOLFIN")

  # Set variables to help find Python library that is compatible with
  # interpreter
  if (PYTHONINTERP_FOUND)
    # Get Python include path from Python interpretter
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
      "import distutils.sysconfig, sys; sys.stdout.write(distutils.sysconfig.get_python_inc())"
      OUTPUT_VARIABLE _PYTHON_INCLUDE_PATH
      RESULT_VARIABLE _PYTHON_INCLUDE_RESULT)

    # Get Python library path from interpreter
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
      "import os, sys, inspect; sys.stdout.write(os.path.split(os.path.split(inspect.getfile(inspect))[0])[0])"
      OUTPUT_VARIABLE _PYTHON_LIB_PATH
      RESULT_VARIABLE _PYTHON_LIB_RESULT)

    # Set include path, if returned by interpreter
    if ("${_PYTHON_INCLUDE_RESULT}" STREQUAL "0")
      set(PYTHON_INCLUDE_DIR ${_PYTHON_INCLUDE_PATH})
    endif()

    # Add a search path for Python library based on output from
    # interpreter
    set(CMAKE_LIBRARY_PATH_SAVE ${CMAKE_LIBRARY_PATH})
    if ("${_PYTHON_LIB_RESULT}" STREQUAL "0")
      set(CMAKE_LIBRARY_PATH ${_PYTHON_LIB_PATH})
    endif()

    # Find Pythons libs
    find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)
    set_package_properties(PythonLibs PROPERTIES TYPE REQUIRED
      DESCRIPTION "Include paths and libraries for Python")
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH_SAVE})
  endif()

  # If Python is found, check for NumPy, SWIG and ply
  if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)

    # Check for NumPy
    find_package(NumPy REQUIRED)
    set_package_properties(NumPy PROPERTIES TYPE REQUIRED
      DESCRIPTION "Array processing for numbers, strings, records, and objects in Python."
      URL "http://www.numpy.org"
      PURPOSE "Needed for the Python interface to DOLFIN")

    # Check for ply
    include(FindPythonModule)
    find_python_module(ply REQUIRED)
    set_package_properties(ply PROPERTIES TYPE REQUIRED
      DESCRIPTION "Python Lex & Yacc"
      URL "http://www.dabeaz.com/ply/"
      PURPOSE "Needed for the Python interface to DOLFIN")
    if (NOT PY_PLY_FOUND)
      message(FATAL_ERROR "Required Python module 'ply' (http://www.dabeaz.com/ply/) could not be found. Install ply or set DOLFIN_ENABLE_PYTHON to false.")
    endif()

    # Check for SWIG
    if (DEFINED UFC_SWIG_EXECUTABLE)
      set(SWIG_EXECUTABLE ${UFC_SWIG_EXECUTABLE})
    endif()
    find_package(SWIG REQUIRED)
    set_package_properties(SWIG PROPERTIES TYPE REQUIRED
      DESCRIPTION "Tool to wrap C/C++ libraries in high-level languages"
      URL "http://swig.org"
      PURPOSE "Needed for the Python interface to DOLFIN")
    set(REQUIRED_SWIG_VERSION "3.0.5")
    if ("${SWIG_VERSION}" VERSION_LESS "${REQUIRED_SWIG_VERSION}")
      message(FATAL_ERROR " DOLFIN requires SWIG version ${REQUIRED_SWIG_VERSION} or greater. You have version ${SWIG_VERSION}. Set DOLFIN_ENABLE_PYTHON to false or install correct SWIG version.")
    endif()
    include(UseSWIG)
    set(PYTHON_FOUND TRUE)
  endif()

endif()

# Check for required package UFC
find_package(UFC MODULE 2017.1)
set_package_properties(UFC PROPERTIES TYPE REQUIRED
  DESCRIPTION "Unified language for form-compilers (part of FFC)"
  URL "https://bitbucket.org/fenics-project/ffc")

# Check for PETSc, SLEPc and petsc4py, slepc4py
set(PETSC_FOUND FALSE)
if (DOLFIN_ENABLE_PETSC)
  find_package(PETSc 3.6)
  set_package_properties(PETSc PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation"
    URL "https://www.mcs.anl.gov/petsc/"
    PURPOSE "Enables the PETSc linear algebra backend")

  if (PETSC_FOUND AND PYTHON_FOUND AND DOLFIN_ENABLE_PETSC4PY)
    find_package(PETSc4py)
    set_package_properties(PETSc4py PROPERTIES TYPE OPTIONAL
      DESCRIPTION "Python bindings for PETSc"
      URL "https://bitbucket.org/petsc/petsc4py/")
    if (PETSC4PY_FOUND)
      if (NOT ("${PETSC4PY_VERSION_MAJOR}" EQUAL "${PETSC_VERSION_MAJOR}"
               AND "${PETSC4PY_VERSION_MINOR}" EQUAL "${PETSC_VERSION_MINOR}") AND NOT DOLFIN_IGNORE_PETSC4PY_VERSION)
        message(WARNING "PETSc version ${PETSC_VERSION} and petsc4py version ${PETSC4PY_VERSION} do not match. Disabling PETSc4py.")
        set(PETSC4PY_FOUND FALSE)
      endif()
    endif()
  endif()

  set(SLEPC_FOUND FALSE)
  if (PETSC_FOUND AND DOLFIN_ENABLE_SLEPC)
    find_package(SLEPc 3.6)
    set_package_properties(SLEPc PROPERTIES TYPE OPTIONAL
      DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
      URL "http://slepc.upv.es/")

    if (SLEPC_FOUND AND PYTHON_FOUND AND DOLFIN_ENABLE_SLEPC4PY)
      find_package(SLEPc4py)
      set_package_properties(SLEPc4py PROPERTIES TYPE OPTIONAL
	DESCRIPTION "Python bindings for SLEPc"
	URL "https://bitbucket.org/slepc/slepc4py/")
      if (SLEPC4PY_FOUND)
        if (NOT ("${SLEPC4PY_VERSION_MAJOR}" EQUAL "${SLEPC_VERSION_MAJOR}"
              AND "${SLEPC4PY_VERSION_MINOR}" EQUAL "${SLEPC_VERSION_MINOR}"))
          message(WARNING "SLEPc version ${SLEPC_VERSION} and slepc4py version ${SLEPC4PY_VERSION} do not match.  Disabling slepc4py support")
          set(SLEPC4PY_FOUND FALSE)
        endif()
      endif()
    endif()
  endif()
endif()

# Check for ParMETIS and SCOTCH
if (DOLFIN_ENABLE_MPI AND MPI_C_FOUND)
  if (DOLFIN_ENABLE_PARMETIS)
    find_package(ParMETIS 4.0.2)
    set_package_properties(ParMETIS PROPERTIES TYPE OPTIONAL
      DESCRIPTION "Parallel Graph Partitioning and Fill-reducing Matrix Ordering"
      URL "http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview"
      PURPOSE "Enables parallel graph partitioning")
  endif()

  if (DOLFIN_ENABLE_SCOTCH)
    find_package(SCOTCH)
    set_package_properties(SCOTCH PROPERTIES TYPE OPTIONAL
      DESCRIPTION "Programs and libraries for graph, mesh and hypergraph partitioning"
      URL "https://www.labri.fr/perso/pelegrin/scotch"
      PURPOSE "Enables parallel graph partitioning")
  endif()
endif()

# Check for UMFPACK
if (DOLFIN_ENABLE_UMFPACK)
  find_package(AMD QUIET)
  find_package(BLAS QUIET)
  set_package_properties(BLAS PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Basic Linear Algebra Subprograms"
    URL "http://netlib.org/blas/")
  find_package(UMFPACK)
  set_package_properties(UMFPACK PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Sparse LU factorization library"
    URL "http://faculty.cse.tamu.edu/davis/suitesparse.html")
endif()

# Check for CHOLMOD
if (DOLFIN_ENABLE_CHOLMOD)
  find_package(CHOLMOD)
  set_package_properties(CHOLMOD PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Sparse Cholesky factorization library for sparse matrices"
    URL "http://faculty.cse.tamu.edu/davis/suitesparse.html")
endif()

# Check for HDF5
if (DOLFIN_ENABLE_HDF5)
  if (NOT DEFINED ENV{HDF5_ROOT})
    set(ENV{HDF5_ROOT} "$ENV{HDF5_DIR}")
  endif()
  set(HDF5_PREFER_PARALLEL TRUE)
  find_package(HDF5 COMPONENTS C)
  set_package_properties(HDF5 PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
    URL "https://www.hdfgroup.org/HDF5")

  # Check that HDF5 has parallel support
  if (HDF5_FOUND)
    if (DOLFIN_ENABLE_MPI)
      if (NOT (HDF5_IS_PARALLEL OR HDF5_ENABLE_PARALLEL))
        message(STATUS "HDF5 has been found, but is missing parallel support. It will not be enabled.")
        set(HDF5_FOUND false)
      endif()
    else()
      message(STATUS "HDF5 has been found, but MPI support is missing. It will not be enabled.")
      set(HDF5_FOUND false)
    endif()
  endif()
endif()

# Check for Trilinos and the requires Trilinos packages
if (DOLFIN_ENABLE_TRILINOS)
  message(STATUS "Checking for Trilinos")
  find_package(Trilinos PATHS ${TRILINOS_DIR} ${Trilinos_DIR}
    $ENV{TRILINOS_DIR})
  set(DOLFIN_TRILINOS_PACKAGES "Tpetra;Zoltan;MueLu;Amesos2;Ifpack2;Belos")

  if ("${Trilinos_VERSION}" VERSION_LESS "12.4.0")
    set(Trilinos_FOUND FALSE)
    message(STATUS "Unable to find Trilinos (>= 12.4.0)")
  endif()

  set_package_properties(Trilinos PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Object-oriented framework for large-scale problems"
    URL "https://trilinos.org")

  # Check for required packages
  set(DOLFIN_TRILINOS_PACKAGES_FOUND false)
  if (Trilinos_FOUND)
    message(STATUS "  Trilinos version ${Trilinos_VERSION} found. Checking for components")

    # Check that necessary packages are enabled
    set(DOLFIN_TRILINOS_PACKAGES_FOUND true)
    foreach (required_package ${DOLFIN_TRILINOS_PACKAGES})

      # Search for required package in list of available packages
      list(FIND Trilinos_PACKAGE_LIST ${required_package} required_trilinos_package_index)
      if(required_trilinos_package_index EQUAL -1)
          set(required_trilinos_package_found false)
      else()
          set(required_trilinos_package_found true)
      endif()

      # Print whether or not package is found
      if (required_trilinos_package_found)
        message(STATUS "    ${required_package} found")
      else()
        message(STATUS "    Trilinos found, but required package ${required_package} not found. Trilinos will be disabled.")
        set(DOLFIN_TRILINOS_PACKAGES_FOUND false)
        break()
      endif()

    endforeach()

    # Add package libraries if all packages have been found
    if (DOLFIN_TRILINOS_PACKAGES_FOUND)
      message(STATUS "  All necessary Trilinos components found. Trilinos will be enabled.")
      set(DOLFIN_TRILINOS_DEFINITIONS)

      # Loop over each package
      foreach (package ${DOLFIN_TRILINOS_PACKAGES})

        # Loop over libs and get full path
        foreach (lib ${${package}_LIBRARIES})
          find_library(TRILINOS_LIB_${lib} ${lib} PATHS ${${package}_LIBRARY_DIRS} NO_DEFAULT_PATH)
          # Also search the default paths
          find_library(TRILINOS_LIB_${lib} ${lib})
          if (TRILINOS_LIB_${lib})
            list(APPEND DOLFIN_TRILINOS_LIBRARIES ${TRILINOS_LIB_${lib}})
          endif()
        endforeach()

      endforeach()

      # Remove duplicates
      list(REVERSE DOLFIN_TRILINOS_LIBRARIES)
      list(REMOVE_DUPLICATES DOLFIN_TRILINOS_LIBRARIES)
      list(REVERSE DOLFIN_TRILINOS_LIBRARIES)

    endif()

  else()
    message(STATUS "Trilinos could not be found")
  endif()
endif()

# Check for zlib
if (DOLFIN_ENABLE_ZLIB)
  find_package(ZLIB)
  set_package_properties(ZLIB PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Compression library"
    URL "http://www.zlib.net")
endif()

# Check for Sphinx
if (DOLFIN_ENABLE_DOCS AND PYTHON_FOUND)
  find_package(Sphinx 1.1.0)
  set_package_properties(Sphinx PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Python documentation generator"
    URL "http://www.sphinx-doc.org"
    PURPOSE "Needed to build documentation")
endif()

# Check for VTK
if (DOLFIN_ENABLE_VTK)
  # Find all modules
  find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} NO_MODULE)
  set_package_properties(VTK PROPERTIES TYPE OPTIONAL
    DESCRIPTION "The Visualization Toolkit - A high level 3D visualization library"
    URL "https://www.vtk.org"
    PURPOSE "Enables plotting with VTK as backend")

  # Add required modules (depends on VTK version)
  if(TARGET vtkRenderingOpenGL2)
    # VTK 7.0
    find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} COMPONENTS vtkViewsCore
      vtkRenderingCore vtkRenderingLabel vtkFiltersTexture vtkRenderingOpenGL2
      NO_MODULE)
  elseif(TARGET vtkRenderingFreeTypeOpenGL)
    # For VTK < 6.3, vtkRenderingFreeTypeOpenGL module is required, so
    # add if available (vtkRenderingFreeTypeOpenGL was removed from
    # VTK 6.3)
    find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} COMPONENTS vtkViewsCore
      vtkRenderingCore vtkRenderingLabel vtkFiltersTexture vtkRenderingOpenGL
      vtkRenderingFreeTypeOpenGL NO_MODULE)
  elseif(TARGET vtkRenderingOpenGL)
    # VTK 6.3
    find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR} COMPONENTS vtkViewsCore
      vtkRenderingCore vtkRenderingLabel vtkFiltersTexture
      vtkRenderingOpenGL NO_MODULE)
  endif()

  set(VTK_VERSION "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
  if (VTK_FOUND)
    if ("${VTK_VERSION}" VERSION_LESS "5.2")
      set(VTK_FOUND FALSE)
      message(WARNING "Unable to find VTK (>= 5.2)")
    else()
      message(STATUS "Found VTK: ${VTK_DIR} (found version \"${VTK_VERSION}\")")
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# Set size of dolfin::la_index

if (PETSC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DDOLFIN_LA_INDEX_SIZE=${PETSC_INT_SIZE}")
else()
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DDOLFIN_LA_INDEX_SIZE=4")
endif()

#------------------------------------------------------------------------------
# Print summary of found and not found optional packages

feature_summary(WHAT ALL)

#------------------------------------------------------------------------------
# Get installation paths for Python modules (pure and platform-dependent)

if (PYTHONINTERP_FOUND)

  if (NOT DEFINED DOLFIN_INSTALL_PYTHON_MODULE_DIR)
    # Get path for platform-dependent Python modules (since we install
    # a binary libary)

    # Python command string to discover module install location
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      set(PYTHON_LIB_DISCOVER_STR "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=True))")
    else()
      set(PYTHON_LIB_DISCOVER_STR "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}'))")
    endif()

    # Probe Python interpreter
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${PYTHON_LIB_DISCOVER_STR}"
      OUTPUT_VARIABLE DOLFIN_INSTALL_PYTHON_MODULE_DIR
      )
    set(DOLFIN_INSTALL_PYTHON_MODULE_DIR ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
      CACHE PATH "Python extension module installation directory.")
  endif()

  if (NOT DEFINED DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR)
    # Get path for pure Python modules

    # Python command string to discover module install location
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      set(PYTHON_LIB_DISCOVER_STR "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=False))")
    else()
      set(PYTHON_LIB_DISCOVER_STR "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=False, prefix='${CMAKE_INSTALL_PREFIX}'))")
    endif()

    # Probe Pyton interpreter
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${PYTHON_LIB_DISCOVER_STR}"
      OUTPUT_VARIABLE DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR
      )
    set(DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR ${DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR}
      CACHE PATH "Python module installation directory.")
  endif()

endif()

#------------------------------------------------------------------------------
# Installation of DOLFIN and FEniCS Python modules

if (DOLFIN_ENABLE_PYTHON AND PYTHON_FOUND)
  install(DIRECTORY
    ${CMAKE_SOURCE_DIR}/site-packages/dolfin
    DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
    USE_SOURCE_PERMISSIONS
    COMPONENT RuntimeLibraries
    PATTERN "*.in" EXCLUDE
    )

  configure_file(${CMAKE_SOURCE_DIR}/site-packages/dolfin/common/globalparameters.py.in
    ${CMAKE_BINARY_DIR}/globalparameters.py @ONLY)

  install(FILES ${CMAKE_BINARY_DIR}/globalparameters.py
    DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin/common/
    COMPONENT RuntimeLibraries
    )

  install(DIRECTORY
    ${CMAKE_SOURCE_DIR}/site-packages/fenics
    DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
    USE_SOURCE_PERMISSIONS
    COMPONENT RuntimeLibraries
    PATTERN "*.in" EXCLUDE
    )
endif()

#------------------------------------------------------------------------------
# Installation of dolfin_utils

if (DOLFIN_INSTALL_PYTHON_MODULE_DIR)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/site-packages/dolfin_utils
    DESTINATION ${DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR}
    USE_SOURCE_PERMISSIONS)

  # Add target "install_dolfin_utils" for installing dolfin_utils
  # without building and install the rest of DOLFIN
  add_custom_target(install_dolfin_utils
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/site-packages/dolfin_utils"
      "${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin_utils"
    COMMENT "Installing dolfin_utils in ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin_utils")
endif()

#------------------------------------------------------------------------------
# Installation of docstrings

#install(DIRECTORY ${CMAKE_SOURCE_DIR}/site-packages/dolfin/docstrings
#        DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin
#        USE_SOURCE_PERMISSIONS)

#------------------------------------------------------------------------------
# Installation of DOLFIN library

# Append the library version information to the library target properties
if (DOLFIN_WITH_LIBRARY_VERSION)
  string(REPLACE "+" "" DOLFIN_LIBRARY_VERSION ${DOLFIN_VERSION})
  # This setting of SOVERSION assumes that any API change
  # will increment either the minor or major version number.
  set(DOLFIN_LIBRARY_PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES}
    VERSION ${DOLFIN_LIBRARY_VERSION}
    SOVERSION ${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}
  )
endif()

# Set DOLFIN install sub-directories
set(DOLFIN_BIN_DIR "bin" CACHE PATH "Binary installation directory.")
set(DOLFIN_LIB_DIR "lib" CACHE PATH "Library installation directory.")
set(DOLFIN_INCLUDE_DIR "include" CACHE PATH "C/C++ header installation directory.")
set(DOLFIN_PKGCONFIG_DIR "lib/pkgconfig" CACHE PATH "pkg-config file installation directory.")
set(DOLFIN_SHARE_DIR "share/dolfin" CACHE PATH "Shared data installation directory.")
set(DOLFIN_MAN_DIR "share/man" CACHE PATH "Manual page installation directory.")
set(DOLFIN_DOC_DIR "${DOLFIN_SHARE_DIR}/doc" CACHE PATH "DOLFIN Documentation directory.")
set(DOLFIN_ETC_DIR "etc" CACHE PATH "Configuration file directory.")

# Add source directory
add_subdirectory(dolfin)

#------------------------------------------------------------------------------
# Installation of DOLFIN utilities

set(DOLFIN_UTILITIES
  ${DOLFIN_SOURCE_DIR}/scripts/dolfin-convert/dolfin-convert
  ${DOLFIN_SOURCE_DIR}/scripts/dolfin-order/dolfin-order
  ${DOLFIN_SOURCE_DIR}/scripts/dolfin-plot/dolfin-plot)

install(FILES ${DOLFIN_UTILITIES}
  DESTINATION ${DOLFIN_BIN_DIR}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
  COMPONENT RuntimeExecutables)

#------------------------------------------------------------------------------
# Installation of DOLFIN manual pages

install(DIRECTORY ${DOLFIN_SOURCE_DIR}/doc-old/man/
  DESTINATION ${DOLFIN_MAN_DIR}
  USE_SOURCE_PERMISSIONS
  COMPONENT RuntimeExecutables)

#------------------------------------------------------------------------------
# Generate and install helper file dolfin.conf

# FIXME: Can CMake provide the library path name variable?
if (APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

# FIXME: not cross-platform compatible
# Create and install dolfin.conf file
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin.conf.in
               ${CMAKE_BINARY_DIR}/dolfin.conf @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfin.conf
        DESTINATION ${DOLFIN_SHARE_DIR}
        COMPONENT Development)

#------------------------------------------------------------------------------
# Generate and install helper file dolfin-version

# FIXME: not cross-platform compatible
# Create and install dolfin-version file
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin-version.in
               ${CMAKE_BINARY_DIR}/dolfin-version @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfin-version
  DESTINATION ${DOLFIN_BIN_DIR}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
  OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
  COMPONENT RuntimeExecutables)

#------------------------------------------------------------------------------
# Generate and install utility script dolfin-get-demos

configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin-get-demos.in
  ${CMAKE_BINARY_DIR}/dolfin-get-demos @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfin-get-demos
  DESTINATION ${DOLFIN_BIN_DIR}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
  OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
  COMPONENT RuntimeExecutables)

#------------------------------------------------------------------------------
# Generate demo files from rst

message(STATUS "")
message(STATUS "Generating demo source files from reStructuredText")
message(STATUS "--------------------------------------------------")
file(GLOB_RECURSE demo_rst_files "*.py.rst" "*.cpp.rst" "*.ufl.rst")
foreach(rst_file ${demo_rst_files})
  execute_process(COMMAND ./utils/pylit/pylit.py ${rst_file}
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR})
  #add_custom_command(TARGET demos_source PRE_BUILD
  #  COMMAND COMMAND ./utils/pylit/pylit.py ${rst_file}
  #  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

#------------------------------------------------------------------------------
# Generate form files for tests, bench, demos and DOLFIN if not exists
# FIXME: Generate files in Build directory instead, at least for
# bench, demo and tests

set(COPY_DEMO_TEST_DEMO_DATA FALSE)
if (NOT EXISTS ${DOLFIN_SOURCE_DIR}/demo/documented/poisson/cpp/Poisson.h)
  message(STATUS "")
  message(STATUS "Generating form files in demo, test and bench directories. May take some time...")
  message(STATUS "----------------------------------------------------------------------------------------")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-form-files.py
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
    RESULT_VARIABLE FORM_GENERATION_RESULT
    OUTPUT_VARIABLE FORM_GENERATION_OUTPUT
    ERROR_VARIABLE FORM_GENERATION_OUTPUT
    )

  if (FORM_GENERATION_RESULT)
    # Cleanup so download is triggered next time we run cmake
    if (EXISTS ${DOLFIN_SOURCE_DIR}/demo/documented/poisson/cpp/Poisson.h)
      file(REMOVE ${DOLFIN_SOURCE_DIR}/demo/documented/poisson/cpp/Poisson.h)
    endif()
    message(FATAL_ERROR "Generation of form files failed: \n${FORM_GENERATION_OUTPUT}")
  endif()
  set(COPY_DEMO_TEST_DEMO_DATA TRUE)
endif()

#------------------------------------------------------------------------------
# Generate demo files from rst

message(STATUS "")
message(STATUS "Generating demo source files")
message(STATUS "--------------------------------------------------")
file(GLOB_RECURSE demo_rst_files "*.py.rst" "*.cpp.rst")
foreach(rst_file ${demo_rst_files})
  execute_process(COMMAND ./utils/pylit/pylit.py ${rst_file}
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR})
endforeach()

#------------------------------------------------------------------------------
# Generate CMakeLists.txt files for bench and demos if not existing

# FIXME: Generate files in Build directory instead?
# NOTE: We need to call this script after generate-formfiles

if (NOT EXISTS ${DOLFIN_SOURCE_DIR}/demo/documented/poisson/cpp/CMakeLists.txt)
  message(STATUS "")
  message(STATUS "Generating CMakeLists.txt files in demo, test and bench directories")
  message(STATUS "-------------------------------------------------------------------")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-cmakefiles.py
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
    RESULT_VARIABLE CMAKE_GENERATION_RESULT
    OUTPUT_VARIABLE CMAKE_GENERATION_OUTPUT
    ERROR_VARIABLE CMAKE_GENERATION_OUTPUT
    )
  if (CMAKE_GENERATION_RESULT)

    # Cleanup so FFC rebuild is triggered next time we run cmake
    if (EXISTS ${DOLFIN_SOURCE_DIR}/demo/documented/poisson/cpp/CMakeLists.txt)
      file(REMOVE ${DOLFIN_SOURCE_DIR}/demo/documented/poisson/cpp/CMakeLists.txt)
    endif()
    message(FATAL_ERROR "Generation of CMakeLists.txt files failed: \n${CMAKE_GENERATION_OUTPUT}")
  endif()

  set(COPY_DEMO_TEST_DEMO_DATA TRUE)
endif()

#------------------------------------------------------------------------------
# Copy data in demo/bench/test direcories to the build directories

# FIXME: We should probably just generate them directly in the build
# directory...

if (COPY_DEMO_TEST_DEMO_DATA)
  message(STATUS "")
  message(STATUS "Copying demo and test data to build directory.")
  message(STATUS "----------------------------------------------")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/copy-test-demo-data.py ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE COPY_DEMO_DATA_RESULT
    OUTPUT_VARIABLE COPY_DEMO_DATA_OUTPUT
    ERROR_VARIABLE COPY_DEMO_DATA_OUTPUT)
  if (COPY_DEMO_DATA_RESULT)
    message(FATAL_ERROR "Copy demo data failed: \n${COPY_DEMO_DATA_OUTPUT}")
  endif()
endif()

#------------------------------------------------------------------------------
# Add demos and install demo source files and mesh files

# Add demo but do not add to default target
message("**** Add demo dir")
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}/modules")
add_subdirectory(demo EXCLUDE_FROM_ALL)
message("**** End Add demo dir")

# Set make program
if ("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
  set(MAKE_PROGRAM "$(MAKE)")
else()
  set(MAKE_PROGRAM "${CMAKE_MAKE_PROGRAM}")
endif()

# Add taget to build .py demo files from Python rst input files, and
# create a target to build source files from .cpp.rst and .py.rst
# files (using pylit)
file(GLOB_RECURSE rst_files "*.py.rst" "*.cpp.rst")
add_custom_target(demos_source)
foreach(rst_file ${rst_files})
  add_custom_command(TARGET demos_source PRE_BUILD
    COMMAND COMMAND ./utils/pylit/pylit.py ${rst_file}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# Add target "demo" for building the demos
add_custom_target(demo
  COMMAND ${MAKE_PROGRAM}
  DEPENDS copy_data_test_demo demos_source
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/demo")

# Install the demo source files
install(DIRECTORY demo DESTINATION ${DOLFIN_SHARE_DIR}
  FILES_MATCHING
  PATTERN "CMakeLists.txt"
  PATTERN "*.cpp"
  PATTERN "*.ufl"
  PATTERN "*.h"
  PATTERN "*.py"
  PATTERN "*.xml*"
  PATTERN "*.off"
  PATTERN "CMakeFiles" EXCLUDE)

 # Install meshes (data directory)
install(DIRECTORY data DESTINATION ${DOLFIN_SHARE_DIR})

#------------------------------------------------------------------------------
# Generate documentation

if (DOLFIN_ENABLE_DOCS)
  if (NOT SPHINX_FOUND)
    message(STATUS "Disabling generation of documentation because Sphinx is missing.")
  else()
    add_subdirectory(doc-old)
  endif()
endif()

#------------------------------------------------------------------------------
# Add tests and benchmarks

if (DOLFIN_ENABLE_BENCHMARKS)
  # Add bench but do not add to default target
  #add_subdirectory(bench EXCLUDE_FROM_ALL)

  # Add target "bench" for building benchmarks
  add_custom_target(bench
    COMMAND ${MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bench")

  # Copy files needed to run benchmarks in build directory
  file(COPY bench DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILES_MATCHING
    PATTERN "*"
    PATTERN "CMakeFiles" EXCLUDE)

  # Add target "run_bench" for running benchmarks
  add_custom_target(run_bench
    COMMAND ${PYTHON_EXECUTABLE} "bench.py"
    DEPENDS bench
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bench")
endif()

if (DOLFIN_ENABLE_TESTING)

  # Google Test needs to be downloaded and compiled by CMake, so we
  # handle it as a special case.
  if (DOLFIN_ENABLE_GTEST)
    # Add sub directory to build GoogleTest
    add_subdirectory(test/unit)

    # Add target "unittests_cpp", but do not add to default target
    add_subdirectory(test/unit/cpp EXCLUDE_FROM_ALL)

    # Add target "run_unittests_cpp" for running only C++ unit tests
    add_custom_target(run_unittests_cpp
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test/unit/cpp/unittests_cpp
      DEPENDS copy_data_test_demo unittests_cpp)
  else()
    # Add dummy targets for C++ unit tests
    add_custom_target(unittests_cpp)
    add_custom_target(run_unittests_cpp)
  endif()

  # FIXME: remove this buildbot updated to call unittests_cpp
  # Add alias for unittests_cpp
  add_custom_target(tests DEPENDS unittests_cpp)

  # Add target "copy_data_test_demo"
  add_custom_target(copy_data_test_demo
    COMMAND ${PYTHON_EXECUTABLE} "-B" ${DOLFIN_SOURCE_DIR}/cmake/scripts/copy-test-demo-data.py ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # Add target "run_memorytests" for running memory tests
  add_custom_target(run_memorytests
    COMMAND ${PYTHON_EXECUTABLE} "-B" "test.py"
    DEPENDS copy_data_test_demo demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/memory")

  # Add target "run_regressiontests" for running regression tests
  add_custom_target(run_regressiontests
    COMMAND ${PYTHON_EXECUTABLE} "-B" "test.py"
    DEPENDS copy_data_test_demo demo demos_source
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/regression")

  # Add target "run_unittests" for running unit tests
  add_custom_target(run_unittests
    DEPENDS run_unittests_cpp run_unittests_py run_unittests_py_mpi)

  # Add target "run_unittests_py" for running Python unit tests
  add_custom_target(run_unittests_py
    COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" "-m" "pytest" "-s"
    DEPENDS copy_data_test_demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/unit/python")

  # Add target "run_unittests_py_mpi" for running Python unit tests with mpi
  add_custom_target(run_unittests_py_mpi
    COMMAND ${MPIEXEC} "-np" "3" "./mpipipe.sh" ${PYTHON_EXECUTABLE} "-B" "-u" "-m" "pytest" "-s"
    DEPENDS copy_data_test_demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/unit/python")

  # Add target "run_quicktest" for running only Python unit tests not
  # marked as 'slow'
  add_custom_target(run_quicktest
    COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" "-m" "pytest" "-s" "-k" "'not" "slow'"
    DEPENDS copy_data_test_demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/unit/python")

  # Add target "run_styletest" for running documentation tests
  add_custom_target(run_styletest
    COMMAND ${PYTHON_EXECUTABLE} "-B" "test_coding_style.py"
    DEPENDS copy_data_test_demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/codingstyle")

  # Create test_coding_style.py in build dir (depends on source path)
  configure_file(${CMAKE_SOURCE_DIR}/test/codingstyle/test_coding_style.py.in
    ${CMAKE_BINARY_DIR}/test/codingstyle/test_coding_style.py @ONLY)

  # Add target "runtests" for running all tests
  add_custom_target(runtests
    DEPENDS run_regressiontests run_unittests run_styletest)
endif()

#------------------------------------------------------------------------------
# Add "make uninstall" target

configure_file(
  "${DOLFIN_CMAKE_DIR}/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

add_custom_target(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

#------------------------------------------------------------------------------
# Print post-install message

add_subdirectory(cmake/post-install)

#------------------------------------------------------------------------------
