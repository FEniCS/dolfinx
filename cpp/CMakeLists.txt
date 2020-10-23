#------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFIN
cmake_minimum_required(VERSION 3.12)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFINX VERSION 2019.2.9.99)

#------------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP000x`

cmake_policy(VERSION 3.12)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

#------------------------------------------------------------------------------
# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Require C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# Get GIT changeset, if available

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
set(CMAKE_MODULE_PATH "${DOLFINX_SOURCE_DIR}/cmake/modules")

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx)

#------------------------------------------------------------------------------
# Configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFIN with shared libraries.")

option(DOLFINX_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages." OFF)
add_feature_info(DOLFINX_SKIP_BUILD_TESTS DOLFINX_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages.")

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

#------------------------------------------------------------------------------
# Enable or disable optional packages

# List optional packages
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")
list(APPEND OPTIONAL_PACKAGES "KaHIP")

# Add options
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "DOLFINX_ENABLE_${OPTIONAL_PACKAGE}" OPTION_NAME)
  option(${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}." ON)
  add_feature_info(${OPTION_NAME} ${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}.")
endforeach()

#------------------------------------------------------------------------------
# Check for MPI

# FIXME: Should we set CMake to use the discovered MPI compiler wrappers?
find_package(MPI 3 REQUIRED)

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
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -pipe)
endif()

# Add some strict compiler checks
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

# Debug flags
CHECK_CXX_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
CHECK_CXX_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -O2)
endif()

#------------------------------------------------------------------------------
# Run tests to find required packages

# Note: When updating Boost version, also update DOLFINXCongif.cmake.in
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED timer filesystem)
set_package_properties(Boost PROPERTIES TYPE REQUIRED
  DESCRIPTION "Boost C++ libraries"
  URL "http://www.boost.org")

# Check for required package Eigen3
find_package(Eigen3 3.2.90 REQUIRED)
set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Lightweight C++ template library for linear algebra"
  URL "http://eigen.tuxfamily.org")

find_package(PETSc 3.10)
set_package_properties(PETSc PROPERTIES TYPE REQUIRED
  DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation (PETSc)"
  URL "https://www.mcs.anl.gov/petsc/"
  PURPOSE "PETSc linear algebra backend")

# Check for HDF5
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)
if (NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR "Found serial HDF5 build, MPI HDF5 build required, try setting HDF5_DIR")
endif()
set_package_properties(HDF5 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
  URL "https://www.hdfgroup.org/HDF5")

find_package(SCOTCH REQUIRED)
set_package_properties(SCOTCH PROPERTIES TYPE REQUIRED
  DESCRIPTION "Programs and libraries for graph, mesh and hypergraph partitioning"
  URL "https://www.labri.fr/perso/pelegrin/scotch"
  PURPOSE "Enables parallel graph partitioning")

# Check for required package UFC
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(UFC MODULE ${DOLFINX_VERSION_MAJOR}.${DOLFINX_VERSION_MINOR})
set_package_properties(UFC PROPERTIES TYPE REQUIRED
  DESCRIPTION "Unified language for form-compilers (part of FFC-X)"
  URL "https://github.com/fenics/ffcx")

  #------------------------------------------------------------------------------
# Run tests to find optional packages

# Check for SLEPc
set(SLEPC_FOUND FALSE)
if (DOLFINX_ENABLE_SLEPC)
  find_package(SLEPc 3.10)
  set_package_properties(SLEPc PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
    URL "http://slepc.upv.es/")
endif()

# Check for ParMETIS
if (DOLFINX_ENABLE_PARMETIS)
  find_package(ParMETIS 4.0.2)
  set_package_properties(ParMETIS PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Parallel Graph Partitioning and Fill-reducing Matrix Ordering"
    URL "http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview"
    PURPOSE "Enables parallel graph partitioning")
endif()

# Check for KaHIP
if (DOLFINX_ENABLE_KAHIP)
  find_package(KaHIP)
  set_package_properties(KaHIP PROPERTIES TYPE OPTIONAL
    DESCRIPTION "A family of graph partitioning programs"
    URL "http://algo2.iti.kit.edu/documents/kahip/index.html"
    PURPOSE "Enables parallel graph partitioning")
endif()

#------------------------------------------------------------------------------
# Print summary of found and not found optional packages

feature_summary(WHAT ALL)

#------------------------------------------------------------------------------
# Installation of DOLFIN library

add_subdirectory(dolfinx)

#------------------------------------------------------------------------------
# Generate and install helper file dolfinx.conf

# FIXME: Can CMake provide the library path name variable?
if (APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

# FIXME: not cross-platform compatible
# Create and install dolfinx.conf file
configure_file(${DOLFINX_SOURCE_DIR}/cmake/templates/dolfinx.conf.in
               ${CMAKE_BINARY_DIR}/dolfinx.conf @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfinx.conf
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx
        COMPONENT Development)

#------------------------------------------------------------------------------
# Copy data in demo/test direcories to the build directories

set(GENERATE_DEMO_TEST_DATA FALSE)
if (Python3_Interpreter_FOUND AND (${DOLFINX_SOURCE_DIR}/demo IS_NEWER_THAN ${CMAKE_CURRENT_BINARY_DIR}/demo OR ${DOLFINX_SOURCE_DIR}/test IS_NEWER_THAN ${CMAKE_CURRENT_BINARY_DIR}/test))
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/demo ${CMAKE_CURRENT_BINARY_DIR}/test)
  set(GENERATE_DEMO_TEST_DATA TRUE)
endif()

if (GENERATE_DEMO_TEST_DATA)
  message(STATUS "")
  message(STATUS "Copying demo and test data to build directory.")
  message(STATUS "----------------------------------------------")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-B" "-u" ${DOLFINX_SOURCE_DIR}/cmake/scripts/copy-test-demo-data.py ${CMAKE_CURRENT_BINARY_DIR} ${PETSC_SCALAR_COMPLEX}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE COPY_DEMO_DATA_RESULT
    OUTPUT_VARIABLE COPY_DEMO_DATA_OUTPUT
    ERROR_VARIABLE COPY_DEMO_DATA_OUTPUT)
  if (COPY_DEMO_DATA_RESULT)
    message(FATAL_ERROR "Copy demo data failed: \n${COPY_DEMO_DATA_OUTPUT}")
  endif()
endif()

#------------------------------------------------------------------------------
# Generate form files for tests and demos

if (GENERATE_DEMO_TEST_DATA)
  message(STATUS "")
  message(STATUS "Generating form files in demo and test directories. May take some time...")
  message(STATUS "----------------------------------------------------------------------------------------")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-B" "-u" ${DOLFINX_SOURCE_DIR}/cmake/scripts/generate-form-files.py ${PETSC_SCALAR_COMPLEX}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE FORM_GENERATION_RESULT
    OUTPUT_VARIABLE FORM_GENERATION_OUTPUT
    ERROR_VARIABLE FORM_GENERATION_OUTPUT
    )

  if (FORM_GENERATION_RESULT)
    # Cleanup so that form generation is triggered next time we run cmake
    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/demo ${CMAKE_CURRENT_BINARY_DIR}/test)
    message(FATAL_ERROR "Generation of form files failed: \n${FORM_GENERATION_OUTPUT}")
  endif()
endif()

#------------------------------------------------------------------------------
# Generate CMakeLists.txt files for demos if needed

# NOTE: We need to call this script after generate-formfiles

if (GENERATE_DEMO_TEST_DATA)
  message(STATUS "")
  message(STATUS "Generating CMakeLists.txt files in demo directory")
  message(STATUS "-------------------------------------------------------------------")
  # Generate CMakeLists.txt files in build directory
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${DOLFINX_SOURCE_DIR}/cmake/scripts/generate-cmakefiles.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE CMAKE_GENERATION_RESULT
    OUTPUT_VARIABLE CMAKE_GENERATION_OUTPUT
    ERROR_VARIABLE CMAKE_GENERATION_OUTPUT
    )
  if (CMAKE_GENERATION_RESULT)
    # Cleanup so FFCX rebuild is triggered next time we run cmake
    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/demo ${CMAKE_CURRENT_BINARY_DIR}/test)
    message(FATAL_ERROR "Generation of CMakeLists.txt files in build directory failed: \n${CMAKE_GENERATION_OUTPUT}")
  else()
    # Generate CMakeLists.txt files in source directory as well, as developers might find it
    # easier to run in-place demo builds while preparing new demos or modifying existing ones
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${DOLFINX_SOURCE_DIR}/cmake/scripts/generate-cmakefiles.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE CMAKE_GENERATION_RESULT
      OUTPUT_VARIABLE CMAKE_GENERATION_OUTPUT
      ERROR_VARIABLE CMAKE_GENERATION_OUTPUT
      )
    if (CMAKE_GENERATION_RESULT)
      message(FATAL_ERROR "Generation of CMakeLists.txt files in source directory failed: \n${CMAKE_GENERATION_OUTPUT}")
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# Install the demo source files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/demo DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx
  FILES_MATCHING
  PATTERN "CMakeLists.txt"
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.c"
  PATTERN "*.cpp"
  PATTERN "*.ufl"
  PATTERN "*.xdmf"
  PATTERN "*.h5"
  PATTERN "CMakeFiles" EXCLUDE)

#------------------------------------------------------------------------------
# Add "make uninstall" target

configure_file(
  "${DOLFINX_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

add_custom_target(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

#------------------------------------------------------------------------------
# Print post-install message

add_subdirectory(cmake/post-install)

#------------------------------------------------------------------------------
