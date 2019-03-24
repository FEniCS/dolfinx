# Top level CMakeLists.txt file for DOLFIN
cmake_minimum_required(VERSION 3.10)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFIN)
set(DOLFIN_VERSION_RELEASE 0)
set(DOLFIN_VERSION_MAJOR "2018")
set(DOLFIN_VERSION_MINOR "2")
set(DOLFIN_VERSION_MICRO "0")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")
if (NOT DOLFIN_VERSION_RELEASE)
  set(DOLFIN_VERSION "${DOLFIN_VERSION}.dev0")
endif()

#------------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP000x`

cmake_policy(VERSION 3.10)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

#------------------------------------------------------------------------------
# Use C++14
set(CMAKE_CXX_STANDARD 14)

# Require C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

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
#option(DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(DOLFIN_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages." OFF)
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)

add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFIN with shared libraries.")
add_feature_info(CMAKE_USE_RELATIVE_PATHS CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects.")
#add_feature_info(DOLFIN_ENABLE_CODE_COVERAGE DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage.")
add_feature_info(DOLFIN_WITH_LIBRARY_VERSION DOLFIN_WITH_LIBRARY_VERSION "Build with library version information.")
add_feature_info(DOLFIN_SKIP_BUILD_TESTS DOLFIN_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages.")

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

#------------------------------------------------------------------------------
# Enable or disable optional packages

# List optional packages
set(OPTIONAL_PACKAGES "")
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "SCOTCH")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")

# Add options
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "DOLFIN_ENABLE_${OPTIONAL_PACKAGE}" OPTION_NAME)
  option(${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}." ON)
  add_feature_info(${OPTION_NAME} ${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}.")
endforeach()

#------------------------------------------------------------------------------
# Package-specific options

# None at the moment

#------------------------------------------------------------------------------
# Check for MPI

# FIXME: Should we set CMake to use the discovered MPI compiler wrappers?
find_package(MPI REQUIRED)

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

# Add flags for generating code coverage reports
#if (DOLFIN_ENABLE_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
#endif()

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
# Note: When updating Boost version, also update FindDOLFIN.cmake.
find_package(Boost 1.56 QUIET REQUIRED)

# Boost public/private libraries to link to.
# Note: These should all be private as they do not appear in the
# DOLFIN public interface , but there is a linking issues with older
# Boost or CMake. Ubuntu 16.04 requires linking DOLFIN programs with
# filesystem, whereas Ubuntu 16.10 and macOS (Homebrew) do not.
if (Boost_VERSION VERSION_LESS 106100)
  set(DOLFIN_BOOST_COMPONENTS_PUBLIC filesystem timer)
  set(DOLFIN_BOOST_COMPONENTS_PRIVATE program_options)
else()
  set(DOLFIN_BOOST_COMPONENTS_PUBLIC timer)
  set(DOLFIN_BOOST_COMPONENTS_PRIVATE filesystem program_options)
endif()

# Find required Boost libraries
find_package(Boost COMPONENTS
  ${DOLFIN_BOOST_COMPONENTS_PUBLIC} ${DOLFIN_BOOST_COMPONENTS_PRIVATE} REQUIRED)
set_package_properties(Boost PROPERTIES TYPE REQUIRED
  DESCRIPTION "Boost C++ libraries"
  URL "http://www.boost.org")

# Check for required package Eigen3
find_package(Eigen3 3.2.90 REQUIRED)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Lightweight C++ template library for linear algebra"
  URL "http://eigen.tuxfamily.org")

find_package(PETSc 3.9)
set_package_properties(PETSc PROPERTIES TYPE REQUIRED
  DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation (PETSc)"
  URL "https://www.mcs.anl.gov/petsc/"
  PURPOSE "PETSc linear algebra backend")

# Check for required package UFC
find_package(UFC MODULE ${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR})
set_package_properties(UFC PROPERTIES TYPE REQUIRED
  DESCRIPTION "Unified language for form-compilers (part of FFC-X)"
  URL "https://github.com/fenics/ffcx")

  # Check for HDF5
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)
if (NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR "Found serial HDF5 build, MPI HDF5 build required, try setting HDF5_DIR")
endif()
set_package_properties(HDF5 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
  URL "https://www.hdfgroup.org/HDF5")

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Check for SLEPc
set(SLEPC_FOUND FALSE)
if (DOLFIN_ENABLE_SLEPC)
  find_package(SLEPc 3.9)
  set_package_properties(SLEPc PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
    URL "http://slepc.upv.es/")
endif()

# Check for ParMETIS and SCOTCH
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

#------------------------------------------------------------------------------
# Print summary of found and not found optional packages

feature_summary(WHAT ALL)

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
include(GNUInstallDirs)
set(DOLFIN_LIB_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Library installation directory.")
set(DOLFIN_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH "C/C++ header installation directory.")
set(DOLFIN_PKGCONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/pkgconfig" CACHE PATH "pkg-config file installation directory.")
set(DOLFIN_SHARE_DIR "${CMAKE_INSTALL_DATAROOTDIR}/dolfin" CACHE PATH "Shared data installation directory.")

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
# Generate form files for tests and demos and DOLFIN if not exists
# FIXME: Generate files in Build directory instead, at least for demo
# and tests

set(COPY_DEMO_TEST_DEMO_DATA FALSE)
if (PYTHONINTERP_FOUND AND NOT EXISTS ${DOLFIN_SOURCE_DIR}/demo/poisson/Poisson.h)
  message(STATUS "")
  message(STATUS "Generating form files in demo and test and directories. May take some time...")
  message(STATUS "----------------------------------------------------------------------------------------")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-form-files.py ${PETSC_SCALAR_COMPLEX}
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
    RESULT_VARIABLE FORM_GENERATION_RESULT
    OUTPUT_VARIABLE FORM_GENERATION_OUTPUT
    ERROR_VARIABLE FORM_GENERATION_OUTPUT
    )

  if (FORM_GENERATION_RESULT)
    # Cleanup so download is triggered next time we run cmake
    if (EXISTS ${DOLFIN_SOURCE_DIR}/demo/poisson/Poisson.h)
      file(REMOVE ${DOLFIN_SOURCE_DIR}/demo/poisson/Poisson.h)
    endif()
    message(FATAL_ERROR "Generation of form files failed: \n${FORM_GENERATION_OUTPUT}")
  endif()
  set(COPY_DEMO_TEST_DEMO_DATA TRUE)
endif()

#------------------------------------------------------------------------------
# Generate CMakeLists.txt files for demos if not existing

# FIXME: Generate files in Build directory instead?
# NOTE: We need to call this script after generate-formfiles

if (PYTHONINTERP_FOUND AND NOT EXISTS ${DOLFIN_SOURCE_DIR}/demo/poisson/CMakeLists.txt)
  message(STATUS "")
  message(STATUS "Generating CMakeLists.txt files in demo and test directories")
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
    if (EXISTS ${DOLFIN_SOURCE_DIR}/demo/poisson/CMakeLists.txt)
      file(REMOVE ${DOLFIN_SOURCE_DIR}/demo/poisson/CMakeLists.txt)
    endif()
    message(FATAL_ERROR "Generation of CMakeLists.txt files failed: \n${CMAKE_GENERATION_OUTPUT}")
  endif()

  set(COPY_DEMO_TEST_DEMO_DATA TRUE)
endif()

#------------------------------------------------------------------------------
# Copy data in demo/test direcories to the build directories

# FIXME: We should probably just generate them directly in the build
# directory...

if (PYTHONINTERP_FOUND AND COPY_DEMO_TEST_DEMO_DATA)
  message(STATUS "")
  message(STATUS "Copying demo and test data to build directory.")
  message(STATUS "----------------------------------------------")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ${DOLFIN_SOURCE_DIR}/cmake/scripts/copy-test-demo-data.py ${CMAKE_CURRENT_BINARY_DIR}
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

# Set make program
if ("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
  set(MAKE_PROGRAM "$(MAKE)")
else()
  set(MAKE_PROGRAM "${CMAKE_MAKE_PROGRAM}")
endif()

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
# Add tests

enable_testing()

# Add target "unittests", but do not add to default target
add_subdirectory(test/unit EXCLUDE_FROM_ALL)

# Add demos but do not add to default target
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}/modules")
add_subdirectory(demo EXCLUDE_FROM_ALL)

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
