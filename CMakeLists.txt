# Top level CMakeLists.txt file for DOLFIN

# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFIN)
set(DOLFIN_VERSION_RELEASE 0)
set(DOLFIN_VERSION_MAJOR "2018")
set(DOLFIN_VERSION_MINOR "1")
set(DOLFIN_VERSION_MICRO "0")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")
if (NOT DOLFIN_VERSION_RELEASE)
  set(DOLFIN_VERSION "${DOLFIN_VERSION}.dev0")
endif()

#------------------------------------------------------------------------------
# Require and use C++14

# Use C++11
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
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfin)

#------------------------------------------------------------------------------
# Configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
option(CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects." OFF)
option(DOLFIN_AUTO_DETECT_MPI "Detect MPI automatically (turn this off to use the MPI compiler wrappers directly via setting CXX, CXX, FC)." ON)
option(DOLFIN_DEPRECATION_ERROR "Turn deprecation warnings into errors." OFF)
#option(DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(DOLFIN_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages." OFF)
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)
#option(DOLFIN_ENABLE_GEOMETRY_DEBUGGING "Enable geometry debugging (developers only; requires libcgal-dev and libcgal-qt5-dev)." OFF)

add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFIN with shared libraries.")
add_feature_info(CMAKE_USE_RELATIVE_PATHS CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects.")
add_feature_info(DOLFIN_AUTO_DETECT_MPI DOLFIN_AUTO_DETECT_MPI "Detect MPI automatically (turn this off to use the MPI compiler wrappers directly via setting CXX, CXX, FC).")
#add_feature_info(DOLFIN_ENABLE_CODE_COVERAGE DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage.")
add_feature_info(DOLFIN_WITH_LIBRARY_VERSION DOLFIN_WITH_LIBRARY_VERSION "Build with library version information.")
add_feature_info(DOLFIN_SKIP_BUILD_TESTS DOLFIN_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages.")
add_feature_info(DOLFIN_DEPRECATION_ERROR DOLFIN_DEPRECATION_ERROR "Turn deprecation warnings into errors.")
#add_feature_info(DOLFIN_ENABLE_GEOMETRY_DEBUGGING DOLFIN_ENABLE_GEOMETRY_DEBUGGING "Enable geometry debugging.")

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
# Enable or disable optional packages

# List optional packages
set(OPTIONAL_PACKAGES "")
list(APPEND OPTIONAL_PACKAGES "MPI")
list(APPEND OPTIONAL_PACKAGES "PETSc")
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "Trilinos")
list(APPEND OPTIONAL_PACKAGES "SCOTCH")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")
list(APPEND OPTIONAL_PACKAGES "zlib")
list(APPEND OPTIONAL_PACKAGES "HDF5")

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

# Settings for Intel compilers
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # Use -isystem incluse flag with Intel compiler
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")

  # Stop spurious warnings from older Intel compilers
  #if("${CMAKE_CXX_COMPILER_VERSION}" LESS "13")
  #  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd654,1125")
  #  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -wd654,1125")
  #  set(CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER} -wd654,1125")
  #  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -wd654,1125")
  #  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -wd654,1125")
  #endif()
endif()

# Set system include flags to get around CMake bug on OSX with gcc See
# http://public.kitware.com/Bug/print_bug_page.php?bug_id=10837
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
#endif()

#if (APPLE)
#  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
#  set(CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER} -Wno-long-long")
#  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-long-long")
#  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wno-long-long")
#endif()

#------------------------------------------------------------------------------
# Check for MPI

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
find_package(Boost 1.56 QUIET REQUIRED)

# Boost public/private libraries to link to.
# Note: These should all be private as they do not appear in the
# DOLFIN public interface , but there is a linking issues with older
# Boost or CMake. Ubuntu 16.04 requires linking DOLFIN programs with
# filesystem, whereas Ubuntu 16.10 and macOS (Homebrew) do not.
if (Boost_VERSION VERSION_LESS 106100)
  set(DOLFIN_BOOST_COMPONENTS_PUBLIC filesystem timer)
  set(DOLFIN_BOOST_COMPONENTS_PRIVATE program_options iostreams)
else()
  set(DOLFIN_BOOST_COMPONENTS_PUBLIC timer)
  set(DOLFIN_BOOST_COMPONENTS_PRIVATE filesystem program_options iostreams)
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

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Is this still needed?
find_package(PythonInterp 3)

set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED
  DESCRIPTION "Interactive high-level object-oriented language"
  URL "http://www.python.org")

# Check for required package UFC
find_package(UFC MODULE 2017.2)
set_package_properties(UFC PROPERTIES TYPE REQUIRED
  DESCRIPTION "Unified language for form-compilers (part of FFC)"
  URL "https://bitbucket.org/fenics-project/ffc")

# Check for PETSc and SLEPc
set(PETSC_FOUND FALSE)
set(SLEPC_FOUND FALSE)
if (DOLFIN_ENABLE_PETSC)
  find_package(PETSc 3.7)
  set_package_properties(PETSc PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation"
    URL "https://www.mcs.anl.gov/petsc/"
    PURPOSE "Enables the PETSc linear algebra backend")

  set(SLEPC_FOUND FALSE)
  if (PETSC_FOUND AND DOLFIN_ENABLE_SLEPC)
    find_package(SLEPc 3.7)
    set_package_properties(SLEPc PROPERTIES TYPE OPTIONAL
      DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
      URL "http://slepc.upv.es/")
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

# Check for HDF5
if (DOLFIN_ENABLE_HDF5)
  if (NOT DEFINED ENV{HDF5_ROOT})
    set(ENV{HDF5_ROOT} "$ENV{HDF5_DIR}")
  endif()
  set(HDF5_PREFER_PARALLEL FALSE)
  if (DOLFIN_ENABLE_MPI)
    set(HDF5_PREFER_PARALLEL TRUE)
  endif()
  find_package(HDF5 COMPONENTS C)
  set_package_properties(HDF5 PROPERTIES TYPE OPTIONAL
    DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
    URL "https://www.hdfgroup.org/HDF5")
endif()

# Check for Trilinos and the requires Trilinos packages
if (DOLFIN_ENABLE_TRILINOS)
  message(STATUS "Checking for Trilinos")
  find_package(Trilinos QUIET PATHS ${TRILINOS_DIR} ${Trilinos_DIR}
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

# Check for geometry debugging
if (DOLFIN_ENABLE_GEOMETRY_DEBUGGING)
  message(STATUS "Enabling geometry debugging")
  find_package(CGAL REQUIRED)
  find_package(GMP REQUIRED)
  find_package(MPFR REQUIRED)
  if (NOT CGAL_FOUND)
    message(FATAL_ERROR "Unable to find package CGAL required for DOLFIN_ENABLE_GEOMETRY_DEBUGGING")
  endif()
  if (NOT GMP_FOUND)
    message(FATAL_ERROR "Unable to find package GMP required for DOLFIN_ENABLE_GEOMETRY_DEBUGGING")
  endif()
  if (NOT MPFR_FOUND)
    message(FATAL_ERROR "Unable to find package MPFR required for DOLFIN_ENABLE_GEOMETRY_DEBUGGING")
  endif()
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
# Generate demo files from rst

if (PYTHONINTERP_FOUND)
  message(STATUS "")
  message(STATUS "Generating demo source files from reStructuredText")
  message(STATUS "--------------------------------------------------")
  file(GLOB_RECURSE demo_rst_files "demo/*.py.rst" "demo/*.cpp.rst" "demo/*.ufl.rst")
  foreach(rst_file ${demo_rst_files})
    execute_process(COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ./utils/pylit/pylit.py ${rst_file}
      WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR})
    #add_custom_command(TARGET demos_source PRE_BUILD
    #  COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ./utils/pylit/pylit.py ${rst_file}
    #  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
endif()

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
    COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-form-files.py
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

# Add taget to build .py demo files from Python rst input files, and
# create a target to build source files from .cpp.rst and .py.rst
# files (using pylit)
if (PYTHONINTERP_FOUND)
  file(GLOB_RECURSE demo_rst_files "demo/*.py.rst" "demo/*.cpp.rst" "demo/*.ufl.rst")
  add_custom_target(demos_source)
  foreach(rst_file ${rst_files})
    add_custom_command(TARGET demos_source PRE_BUILD
      COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ./utils/pylit/pylit.py ${rst_file}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
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
  PATTERN "*.off"
  PATTERN "CMakeFiles" EXCLUDE)

 # Install meshes (data directory)
install(DIRECTORY data DESTINATION ${DOLFIN_SHARE_DIR})

#------------------------------------------------------------------------------
# Add tests and benchmarks

enable_testing()

# Add target "unittests", but do not add to default target
add_subdirectory(test/unit/cpp EXCLUDE_FROM_ALL)

# Add demo but do not add to default target
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}/modules")
add_subdirectory(demo EXCLUDE_FROM_ALL)

# Add target "copy_data_test_demo"
add_custom_target(copy_data_test_demo
  COMMAND ${PYTHON_EXECUTABLE} "-B" "-u" ${DOLFIN_SOURCE_DIR}/cmake/scripts/copy-test-demo-data.py ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Add target "demos" for building the demos
add_custom_target(demos
  COMMAND ${MAKE_PROGRAM}
  DEPENDS copy_data_test_demo demos_source
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/demo")

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
