#------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFINx
cmake_minimum_required(VERSION 3.16)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFINX VERSION "0.4.2.0")

set(DOXYGEN_DOLFINX_VERSION ${DOLFINX_VERSION} CACHE STRING "Version for Doxygen" FORCE)

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

option(BUILD_SHARED_LIBS "Build DOLFINx with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINx with shared libraries.")

option(DOLFINX_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages." OFF)
add_feature_info(DOLFINX_SKIP_BUILD_TESTS DOLFINX_SKIP_BUILD_TESTS "Skip build tests for testing usability of dependency packages.")

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# Enable SIMD with xtensor
option(XTENSOR_USE_XSIMD "Enable SIMD with xtensor." OFF)
add_feature_info(XTENSOR_USE_XSIMD XTENSOR_USE_XSIMD "Enable SIMD with xtensor (xsimd).")

# Enable xtensor with target-specific optimization, i.e. -march=native
option(XTENSOR_OPTIMIZE "Enable xtensor target-specific optimization" OFF)
add_feature_info(XTENSOR_OPTIMIZE XTENSOR_OPTIMIZE "Enable architecture-specific optimizations as defined by xtensor.")

#------------------------------------------------------------------------------
# Enable or disable optional packages

# List optional packages
list(APPEND OPTIONAL_PACKAGES "ADIOS2")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")
list(APPEND OPTIONAL_PACKAGES "SCOTCH")
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "KaHIP")

# Add options
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "DOLFINX_ENABLE_${OPTIONAL_PACKAGE}" OPTION_NAME)
  option(${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}." ON)
  add_feature_info(${OPTION_NAME} ${OPTION_NAME} "Compile with support for ${OPTIONAL_PACKAGE}.")
endforeach()

#------------------------------------------------------------------------------
# Check for MPI

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
# Find required packages

# Note: When updating Boost version, also update DOLFINXCongif.cmake.in
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED timer)
set_package_properties(Boost PROPERTIES TYPE REQUIRED
  DESCRIPTION "Boost C++ libraries"
  URL "http://www.boost.org")

# Use Python for detecting UFCx and Basix
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Check for Basix
# Note: Basix may be installed as a standalone C++ library, or in the
# Basix Python module tree
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if (BASIX_PY_DIR)
  message(STATUS "Adding ${BASIX_PY_DIR} to Basix/xtensor/xtl search hints")
  set(xtl_ROOT "${BASIX_PY_DIR};${xtl_ROOT}")
  set(xsimd "${BASIX_PY_DIR};${xsimd_ROOT}")
  set(xtensor_ROOT "${BASIX_PY_DIR};${xtensor_ROOT}")

  # Basix installed from manylinux wheel
  if (IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
    set(CMAKE_INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
  endif()
endif()
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})
set_package_properties(basix PROPERTIES TYPE REQUIRED
  DESCRIPTION "FEniCS tabulation library"
  URL "https://github.com/fenics/basix")

get_target_property(BASIX_DEFN Basix::basix INTERFACE_COMPILE_DEFINITIONS)
if("XTENSOR_USE_XSIMD" IN_LIST BASIX_DEFN)
  find_package(xsimd REQUIRED)
endif()

# Check for xtensor
find_package(xtensor 0.23.10 REQUIRED)
set_package_properties(xtensor PROPERTIES TYPE REQUIRED
  DESCRIPTION "C++ library for numerical analysis with multi-dimensional array expressions."
  URL "https://xtensor.readthedocs.io/")

# Check for PETSc
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)

# Check if PETSc build uses real or complex scalars (this is configured
# in DOLFINxConfig.cmake.in)
include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
check_symbol_exists(PETSC_USE_COMPLEX petscsystypes.h HAVE_PETSC_SCALAR_COMPLEX)

# Setting for FeatureSummary
if(PETSC_FOUND)
  set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND PETSc)
else()
  set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND PETSc)
endif()
set_package_properties(PETSc PROPERTIES TYPE REQUIRED
  DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation (PETSc)"
  URL "https://www.mcs.anl.gov/petsc/"
  PURPOSE "PETSc linear algebra backend")

# Check for HDF5
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_FIND_DEBUG TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
if (NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR "Found serial HDF5 build, MPI HDF5 build required, try setting HDF5_DIR or HDF5_ROOT")
endif()
set_package_properties(HDF5 PROPERTIES TYPE REQUIRED
  DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
  URL "https://www.hdfgroup.org/HDF5")

# Check for UFC. First check for UFCxConfig.cmake (CONFIG)
# Note: we use the case (ufcx vs UFCx) elsewhere to determine by which
# method UFCx was found
find_package(ufcx ${DOLFINX_VERSION_MAJOR}.${DOLFINX_VERSION_MINOR} CONFIG)
if (NOT ufcx_FOUND)
  # Next, check in MODULE mode (using FindUFCX.cmake)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  find_package(UFCx ${DOLFINX_VERSION_MAJOR}.${DOLFINX_VERSION_MINOR} REQUIRED MODULE)
endif()
set_package_properties(UFCx PROPERTIES TYPE REQUIRED
  DESCRIPTION "Interface for form-compilers (part of FFCx)"
  URL "https://github.com/fenics/ffcx")

#------------------------------------------------------------------------------
# Find optional packages

if (DOLFINX_ENABLE_ADIOS2)
  find_package(ADIOS2 2.7.1)
endif()
set_package_properties(ADIOS2 PROPERTIES TYPE OPTIONAL
  DESCRIPTION "Adaptable Input/Output (I/O) System."
  URL "https://adios2.readthedocs.io/en/latest/"
  PURPOSE "IO, including in parallel")

if (DOLFINX_ENABLE_SLEPC)
  find_package(PkgConfig REQUIRED)
  set(ENV{PKG_CONFIG_PATH} "$ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{SLEPC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}:$ENV{PETSC_DIR}:$ENV{PKG_CONFIG_PATH}")
  pkg_search_module(SLEPC IMPORTED_TARGET slepc>=3.15)

  # Setting for FeatureSummary
  if(SLEPC_FOUND)
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND SLEPc)
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND SLEPc)
  endif()
endif()
set_package_properties(SLEPc PROPERTIES TYPE RECOMMENDED
  DESCRIPTION "Scalable Library for Eigenvalue Problem Computations"
  URL "http://slepc.upv.es/"
  PURPOSE "Eigenvalue computation")

if (DOLFINX_ENABLE_SCOTCH)
  find_package(SCOTCH)
endif()
set_package_properties(SCOTCH PROPERTIES TYPE OPTIONAL
  DESCRIPTION "Programs and libraries for graph, mesh and hypergraph partitioning"
  URL "https://www.labri.fr/perso/pelegrin/scotch"
  PURPOSE "Parallel graph partitioning")

if (DOLFINX_ENABLE_PARMETIS)
  find_package(ParMETIS 4.0.2)
endif()
set_package_properties(ParMETIS PROPERTIES TYPE RECOMMENDED
  DESCRIPTION "Parallel Graph Partitioning and Fill-reducing Matrix Ordering"
  URL "http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview"
  PURPOSE "Parallel graph partitioning")

if (DOLFINX_ENABLE_KAHIP)
  find_package(KaHIP)
endif()
set_package_properties(KaHIP PROPERTIES TYPE OPTIONAL
  DESCRIPTION "A family of graph partitioning programs"
  URL "https://kahip.github.io/"
  PURPOSE "Parallel graph partitioning")

# Check that at least one graph partitioner has been found
if (NOT SCOTCH_FOUND AND NOT PARMETIS_FOUND AND NOT KAHIP_FOUND)
  message(FATAL_ERROR "No graph partitioner found. SCOTCH, ParMETIS or KaHIP is required.")
endif()

#------------------------------------------------------------------------------
# Print summary of found and not found optional packages

feature_summary(WHAT ALL)

#------------------------------------------------------------------------------
# Installation of DOLFINx library

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
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/dolfinx
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
# Install the demo source files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/demo DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx
  FILES_MATCHING
  PATTERN "CMakeLists.txt"
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.c"
  PATTERN "*.cpp"
  PATTERN "*.py"
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
