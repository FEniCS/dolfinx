# Top level CMakeLists.txt file for DOLFIN

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFIN)
set(DOLFIN_VERSION_MAJOR "0")
set(DOLFIN_VERSION_MINOR "9")
set(DOLFIN_VERSION_MICRO "8")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")
add_definitions(-DPACKAGE_VERSION="${DOLFIN_VERSION}")

#------------------------------------------------------------------------------
# General configuration

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Set special link option, see `cmake --help-policy CMP0003`
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

# Set location of our FindFoo.cmake modules
set(DOLFIN_CMAKE_DIR "${DOLFIN_SOURCE_DIR}/cmake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}" ${CMAKE_MODULE_PATH})

#------------------------------------------------------------------------------
# Configurable options for how we want to build

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
#option(CMAKE_SKIP_RPATH "Do not add runtime paths when using shared libraries." ON)
option(CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects." OFF)
option(DOLFIN_DEBUG_UBLAS "Add some extra uBLAS debug information." OFF)
#option(DOLFIN_ENABLE_DOCS "Build documentation." OFF)
option(DOLFIN_ENABLE_UNIT_TESTS "Enable unit tests." ON)
option(DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)

#------------------------------------------------------------------------------
# Enable or disable optional packages

# FIXME: Discuss which should be enabled by default
# FIXME: AL: I suggest only the Python wrappers
# FIXME: GNW: That would lead to most demos failing (no LU solver)
# FIXME: GNW: I suggest enabling everything, but using QUIET to eliminate
#             excessive output
# FIXME: JR: What about adopting the way they do this in Trilinos? Then we
#            could have options like DOLFIN_ENABLE_ALL_PACKAGES for enabling
#            everything and DOLFIN_ENABLE_DEFAULT_PACKAGES for enabling only
#            some default packages.

option(DOLFIN_ENABLE_MPI "Compile with support for MPI." ON)
option(DOLFIN_ENABLE_PETSC "Compile with support for PETSc linear algebra." ON)
option(DOLFIN_ENABLE_SLEPC "Compile with support for SLEPc." ON)
option(DOLFIN_ENABLE_SCOTCH "Compile with support for SCOTCH graph partitioning." ON)
option(DOLFIN_ENABLE_UMFPACK "Compile with support for UMFPACK." ON)
option(DOLFIN_ENABLE_TRILINOS "Compile with support for Trilinos." ON)
option(DOLFIN_ENABLE_CHOLMOD "Compile with support for CHOLMOD." ON)
option(DOLFIN_ENABLE_MTL4 "Compile with support for MTL4." ON)
option(DOLFIN_ENABLE_PARMETIS "Compile with support for ParMETIS." ON)
option(DOLFIN_ENABLE_GMP "Compile with support for GMP." ON)
option(DOLFIN_ENABLE_ZLIB "Compile with support for zlib." ON)
option(DOLFIN_ENABLE_CGAL "Compile with support for CGAL" ON)
option(DOLFIN_ENABLE_PYTHON "Compile the Python wrappers." ON)

#------------------------------------------------------------------------------
# Package-specific options

option(CGAL_DISABLE_ROUNDING_MATH_CHECK "Disable rounding math check in CGAL. This permits Valgrind to run." OFF)

set(DOLFIN_BOOST_USE_MULTITHREADED OFF CACHE BOOL "Use multithreaded Boost libraries.")

#------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo Maintainer MinSizeRel." FORCE)
endif()

# 'Maintainer' build type flags
set(CMAKE_CXX_FLAGS_MAINTAINER "-O2 -g -ansi -pipe -Wall -Werror -DDEBUG" CACHE STRING
  "Flags used by the compiler during development." FORCE)

# FIXME: Do we want to add -DDEBUG to RelWithDebInfo?

# Do not debug uBLAS unless requested
if (NOT DOLFIN_DEBUG_UBLAS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_UBLAS_NDEBUG")
endif()

#------------------------------------------------------------------------------
# Run tests to find required packages

find_package(UFC.dolfin  REQUIRED)
find_package(Armadillo.dolfin REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(Boost.dolfin REQUIRED)

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Check for Python
if (DOLFIN_ENABLE_PYTHON)
  find_package(PythonInterp QUIET)
  find_package(PythonLibs QUIET)

  # If Python is found, check for NumPy and SWIG
  if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    find_package(NumPy.dolfin REQUIRED)
    find_package(SWIG REQUIRED)
    include(UseSWIG)
  endif()
  message(STATUS "Python lib dir ${PYTHON_LIBRARIES}")
endif()

# Check for MPI
if (DOLFIN_ENABLE_MPI)
  find_package(MPI QUIET)
  if (DOLFIN_ENABLE_PARMETIS)
    find_package(ParMETIS.dolfin QUIET)
  endif()
endif()

# Check for SCOTCH
if (DOLFIN_ENABLE_SCOTCH)
  find_package(ScotchPT QUIET)
endif()

# Check for UMFPACK
if (DOLFIN_ENABLE_UMFPACK)
  find_package(UMFPACK.dolfin QUIET)
endif()

# Check for CHOLMOD
if (DOLFIN_ENABLE_CHOLMOD)
  find_package(Cholmod QUIET)
endif()

# Check for Trilinos
if (DOLFIN_ENABLE_TRILINOS)
  find_package(Trilinos PATHS $ENV{TRILINOS_DIR}/include QUIET)
endif()

# Check for MTL4
if (DOLFIN_ENABLE_MTL4)
  find_package(MTL4.dolfin QUIET)
endif()

# Check for PETSc
if (DOLFIN_ENABLE_PETSC)
  find_package(PETSc.dolfin QUIET)
  if (PETSC_FOUND AND DOLFIN_ENABLE_SLEPC)
    find_package(SLEPc QUIET)
  endif()
endif()

# Check for GMP
if (DOLFIN_ENABLE_GMP)
  find_package(GMP QUIET)
endif()

# Check for zlib
if (DOLFIN_ENABLE_ZLIB)
  find_package(ZLIB QUIET)
endif()

# Check for CGAL
if (DOLFIN_ENABLE_CGAL)
  find_package(CGAL PATHS $ENV{CGAL_DIR}/lib QUIET)
endif()

# Check for cppunit
if (DOLFIN_ENABLE_UNIT_TESTS)
  find_package(CppUnit.dolfin QUIET)
endif()

#------------------------------------------------------------------------------
# Installation of Python module (if enabled)

if (PYTHONINTERP_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}'))"
    OUTPUT_VARIABLE PYTHON_INSTALL_PATH
    )

  set(DOLFIN_INSTALL_PYTHON_MODULE_DIR ${PYTHON_INSTALL_PATH})
  set(DOLFIN_INSTALL_PYTHON_EXT_DIR ${PYTHON_INSTALL_PATH})
endif()

#------------------------------------------------------------------------------
# Installation of dolfin_utils

install(DIRECTORY ${CMAKE_SOURCE_DIR}/site-packages/dolfin_utils
        DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
        USE_SOURCE_PERMISSIONS)

#------------------------------------------------------------------------------
# Installation of DOLFIN library

# Append the library version information to the library target properties
if (DOLFIN_WITH_LIBRARY_VERSION)
  set(DOLFIN_LIBRARY_PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES}
    VERSION ${DOLFIN_VERSION}
    SOVERSION ${DOLFIN_VERSION_MAJOR}
  )
endif()

# Set DOLFIN install sub-directories
set(DOLFIN_BIN_DIR "bin")
set(DOLFIN_LIB_DIR "lib")
set(DOLFIN_INCLUDE_DIR "include")
set(DOLFIN_PKGCONFIG_DIR "lib/pkgconfig")
set(DOLFIN_SHARE_DIR "share")
set(DOLFIN_MAN_DIR "share/man")

# Add source directory
add_subdirectory(dolfin)

#------------------------------------------------------------------------------
# Generate helper file dolfin.conf

# FIXME: not cross-platform compatible
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin.conf.in ${CMAKE_CURRENT_BINARY_DIR}/dolfin.conf @ONLY)
