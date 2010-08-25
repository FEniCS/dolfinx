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
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}/modules")

#------------------------------------------------------------------------------
# Configurable options for how we want to build

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
#option(CMAKE_SKIP_RPATH "Do not add runtime paths when using shared libraries." ON)
option(CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects." OFF)
option(DOLFIN_DEBUG_UBLAS "Use extra uBLAS debugging." OFF)
option(DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)

#option(DOLFIN_ENABLE_DOCS "Build documentation." OFF)
#option(CMAKE_SKIP_RPATH "Do not add runtime paths when using shared libraries." ON)

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

# List optional packages
set(OPTIONAL_PACKAGES "")
list(APPEND OPTIONAL_PACKAGES "MPI")
list(APPEND OPTIONAL_PACKAGES "PETSc")
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "Trilinos")
list(APPEND OPTIONAL_PACKAGES "MTL4")
list(APPEND OPTIONAL_PACKAGES "UMFPACK")
list(APPEND OPTIONAL_PACKAGES "CHOLMOD")
list(APPEND OPTIONAL_PACKAGES "SCOTCH")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")
list(APPEND OPTIONAL_PACKAGES "GMP")
list(APPEND OPTIONAL_PACKAGES "CGAL")
list(APPEND OPTIONAL_PACKAGES "zlib")
list(APPEND OPTIONAL_PACKAGES "Python")

# Add options
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "DOLFIN_ENABLE_${OPTIONAL_PACKAGE}" OPTION_NAME)
  option(${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}." ON)
endforeach()

#------------------------------------------------------------------------------
# Package-specific options

option(CGAL_DISABLE_ROUNDING_MATH_CHECK "Disable rounding math check in CGAL. This permits Valgrind to run." OFF)

#------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo Maintainer MinSizeRel." FORCE)
endif()

# Check for some compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-pipe HAVE_PIPE)
if (HAVE_PIPE)
  set(DOLFIN_CXX_RELEASE_FLAGS "-pipe ${DOLFIN_CXX_RELEASE_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG("-Wall -Werror" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  set(DOLFIN_CXX_RELEASE_FLAGS "-Wall -Werror ${DOLFIN_CXX_RELEASE_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-std=c++98 HAVE_STD)
if (HAVE_STD)
  set(DOLFIN_CXX_RELEASE_FLAGS "-std=c++98 ${DOLFIN_CXX_RELEASE_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  set(DOLFIN_CXX_RELEASE_FLAGS "-g ${DOLFIN_CXX_RELEASE_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  set(DOLFIN_CXX_RELEASE_FLAGS "-O2 ${DOLFIN_CXX_RELEASE_FLAGS}")
endif()

# Set 'Maintainer' build type flags
set(CMAKE_CXX_FLAGS_MAINTAINER "${DOLFIN_CXX_RELEASE_FLAGS} -DDEBUG" CACHE STRING
  "Flags used by the compiler during development." FORCE)

# FIXME: Do we want to add -DDEBUG to RelWithDebInfo?

# Do not debug uBLAS unless requested
if (NOT DOLFIN_DEBUG_UBLAS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_UBLAS_NDEBUG")
endif()

#------------------------------------------------------------------------------
# Run tests to find required packages

# FIXME: Should Boost_USE_STATIC_LIBS and Boost_USE_MULTITHREADED be options?

# Check for Boost
set(Boost_USE_STATIC_LIBS   OFF)
set(Boost_USE_MULTITHREADED OFF)
set(BOOST_ROOT $ENV{BOOST_DIR})
set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0)
find_package(Boost 1.36 COMPONENTS filesystem program_options system REQUIRED)

# Check for required packages
find_package(UFC PATHS $ENV{CMAKE_CONFIG_PATH}/share/ufc/cmake REQUIRED)
find_package(Armadillo REQUIRED)
find_package(LibXml2 REQUIRED)

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Note: Check for Python interpreter even when Python is disable because it is 
#       used to get the installation path for dolfin_utils 
find_package(PythonInterp)
if (DOLFIN_ENABLE_PYTHON)
  find_package(PythonLibs)

  # If Python is found, check for NumPy and SWIG
  if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    find_package(NumPy REQUIRED)
    find_package(SWIG REQUIRED)
    include(UseSWIG)
    set(PYTHON_FOUND TRUE)
  endif()

endif()

# FIXME: Should we move the MPI check to the compiler section?
# Check for MPI, ParMETIS and SCOTCH
if (DOLFIN_ENABLE_MPI)
  find_package(MPI)

  # FIXME: Should be set CMake to use the MPI compiler wrappers?

  # Check for ParMETIS
  if (DOLFIN_ENABLE_PARMETIS)
    find_package(ParMETIS)
  endif()

  # Check for SCOTCH
  if (DOLFIN_ENABLE_SCOTCH)
    find_package(ScotchPT)
  endif()

endif()

# Check for UMFPACK
if (DOLFIN_ENABLE_UMFPACK)
  find_package(AMD)
  find_package(BLAS)
  find_package(UMFPACK)
endif()

# Check for CHOLMOD
if (DOLFIN_ENABLE_CHOLMOD)
  find_package(CHOLMOD)
endif()

# Check for Trilinos
if (DOLFIN_ENABLE_TRILINOS)
  find_package(Trilinos PATHS $ENV{TRILINOS_DIR}/include QUIET)
endif()

# Check for MTL4
if (DOLFIN_ENABLE_MTL4)
  find_package(MTL4)
endif()

# Check for PETSc and SLEPc
if (DOLFIN_ENABLE_PETSC)
  find_package(PETSc)
  if (PETSC_FOUND AND DOLFIN_ENABLE_SLEPC)
    find_package(SLEPc)
  endif()
endif()

# FIXME: Check for GMPXX (C++ bindings) too? CGAL seems to need it
# Check for GMP
if (DOLFIN_ENABLE_GMP)
  find_package(GMP)
endif()

# Check for CGAL
if (DOLFIN_ENABLE_CGAL)
  find_package(CGAL)
endif()

# Check for zlib
if (DOLFIN_ENABLE_ZLIB)
  find_package(ZLIB)
endif()

# Check for cppunit
if (DOLFIN_ENABLE_UNIT_TESTS)
  find_package(CppUnit)
endif()

# Gather information about which optional packages were found and not found
set(OPTIONAL_PACKAGES_NOT_FOUND "")
set(OPTIONAL_PACKAGES_FOUND "")
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
  if (${PKG}_FOUND OR ${OPTIONAL_PACKAGE}_FOUND)
    list(APPEND OPTIONAL_PACKAGES_FOUND ${PKG})
  else()
    list(APPEND OPTIONAL_PACKAGES_NOT_FOUND ${PKG})
  endif()
endforeach()

# Print information about any found optional packages
if (OPTIONAL_PACKAGES_FOUND)
  message(STATUS "The following optional packages were found:")
  foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES_FOUND})
    string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
    message(STATUS "(OK) ${OPTIONAL_PACKAGE}")
  endforeach()
endif()

# Print information about any optional packages that were not found
if (OPTIONAL_PACKAGES_NOT_FOUND)
  message(STATUS "The following optional packages could not be found:")
  foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES_NOT_FOUND})
    string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
    message(STATUS "(**) ${OPTIONAL_PACKAGE}")
  endforeach()
endif()

#------------------------------------------------------------------------------
# Get installation paths for Python modules (pure and platform-dependent)

if (PYTHONINTERP_FOUND)

  # Get path for platform-dependent Python modules (since we install a binary libary)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}'))"
    OUTPUT_VARIABLE DOLFIN_INSTALL_PYTHON_MODULE_DIR
    )

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=False, prefix='${CMAKE_INSTALL_PREFIX}'))"
    OUTPUT_VARIABLE DOLFIN_PYTHON_PURE_PACKAGE_PATH
    )
  
endif()

#------------------------------------------------------------------------------
# Installation of dolfin_utils

if (DOLFIN_INSTALL_PYTHON_MODULE_DIR)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/site-packages/dolfin_utils
    DESTINATION ${DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR}
    USE_SOURCE_PERMISSIONS)
endif()

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
set(DOLFIN_SHARE_DIR "share/dolfin")
set(DOLFIN_MAN_DIR "share/man")

# Add source directory
add_subdirectory(dolfin)

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
# Build and install DOLFIN demos

# Add demos but do not add to default target
add_subdirectory(demo EXCLUDE_FROM_ALL)

# Install the demo source files
install(DIRECTORY demo DESTINATION ${DOLFIN_SHARE_DIR}
        FILES_MATCHING
        PATTERN "CMakeLists.txt"
        PATTERN "*.cpp"
        PATTERN "*.ufl"
        PATTERN "*.h"
        PATTERN "*.py"
        PATTERN "*.xml*"
        PATTERN "CMakeFiles" EXCLUDE)

#------------------------------------------------------------------------------
# Build and install DOLFIN tests

# Add tests but do not add to default target
add_subdirectory(test EXCLUDE_FROM_ALL)
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo
#        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/dolfin
#        USE_SOURCE_PERMISSIONS)

#------------------------------------------------------------------------------
