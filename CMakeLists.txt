# Top level CMakeLists.txt file for DOLFIN

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# Set project name and version number

project(DOLFIN)
set(DOLFIN_VERSION_MAJOR "1")
set(DOLFIN_VERSION_MINOR "0")
set(DOLFIN_VERSION_MICRO "0+")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")

#------------------------------------------------------------------------------
# General configuration

# Set CMake options, see `cmake --help-policy CMP000x`
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0004 OLD)
endif()

# Set location of our FindFoo.cmake modules
set(DOLFIN_CMAKE_DIR "${DOLFIN_SOURCE_DIR}/cmake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}/modules")

# Make sure CMake uses the correct dolfin-config.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfin)

#------------------------------------------------------------------------------
# Configurable options for how we want to build

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
option(CMAKE_SKIP_RPATH "Do not add runtime paths when using shared libraries." OFF)
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
option(CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects." OFF)
option(DOLFIN_DEBUG_UBLAS "Use extra uBLAS debugging." OFF)
option(DOLFIN_ENABLE_CODE_COVERAGE "Enable code coverage." OFF)
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)
option(DOLFIN_ENABLE_UNIT_TESTS "Enable unit tests." ON)
option(DOLFIN_ENABLE_TESTING "Enable testing." OFF)
option(DOLFIN_ENABLE_BENCHMARKS "Enable benchmark programs." OFF)
option(DOLFIN_ENABLE_DOCS "Enable generation of documentation." ON)

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
list(APPEND OPTIONAL_PACKAGES "OpenMP")
list(APPEND OPTIONAL_PACKAGES "MPI")
list(APPEND OPTIONAL_PACKAGES "PETSc")
list(APPEND OPTIONAL_PACKAGES "SLEPc")
list(APPEND OPTIONAL_PACKAGES "Trilinos")
list(APPEND OPTIONAL_PACKAGES "MTL4")
list(APPEND OPTIONAL_PACKAGES "UMFPACK")
list(APPEND OPTIONAL_PACKAGES "CHOLMOD")
list(APPEND OPTIONAL_PACKAGES "PaStiX")
list(APPEND OPTIONAL_PACKAGES "SCOTCH")
list(APPEND OPTIONAL_PACKAGES "ParMETIS")
list(APPEND OPTIONAL_PACKAGES "CGAL")
list(APPEND OPTIONAL_PACKAGES "zlib")
list(APPEND OPTIONAL_PACKAGES "Python")
list(APPEND OPTIONAL_PACKAGES "Sphinx")
list(APPEND OPTIONAL_PACKAGES "VTK")

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

# Disable strict checks that fail for inlcuded 3rd party libraries
CHECK_CXX_COMPILER_FLAG(-Wno-long-long HAVE_DISABLE_PEDANTIC_CHECKS)
if (HAVE_DISABLE_PEDANTIC_CHECKS)
  set(DOLFIN_CXX_DEVELOPER_FLAGS "-Wno-long-long ${DOLFIN_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-std=c++98 HAVE_STD)
if (HAVE_STD AND CMAKE_COMPILER_IS_GNUCXX)
  set(DOLFIN_CXX_DEVELOPER_FLAGS "-std=c++98 ${DOLFIN_CXX_DEVELOPER_FLAGS}")
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

# Do not debug uBLAS unless requested
if (NOT DOLFIN_DEBUG_UBLAS)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DBOOST_UBLAS_NDEBUG")
endif()

# FIXME: Do we want to add -DDEBUG to RelWithDebInfo?

# Add debug definitions
if (CMAKE_BUILD_TYPE STREQUAL "Developer" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DDEBUG")
endif()

# Add flags for generating code coverage reports
if (DOLFIN_ENABLE_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# Disable some warnings for Intel compilers
if(${CMAKE_CXX_COMPILER} MATCHES "icpc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd654,1125")
endif()

#------------------------------------------------------------------------------
# Enable Fortran if available. Used for some configuration tests (e.g. BLAS)

# This workaround address a CMake bug (http://cmake.org/Bug/view.php?id=9220)
include(language_support_v2)
workaround_9220(Fortran Fortran_language_works)
if (Fortran_language_works)
 enable_language(Fortran)
endif()

#------------------------------------------------------------------------------
# Check for MPI and OpenMP

# FIXME: Should be set CMake to use the MPI compiler wrappers?

if (DOLFIN_ENABLE_MPI)
  find_package(MPI)
endif()

if (DOLFIN_ENABLE_OPENMP)
  find_package(OpenMP)
  include(CheckOpenMP)
  check_openmp_unsigned_int_loop_control_variable(OPENMP_UINT_TEST_RUNS)
  if (NOT OPENMP_UINT_TEST_RUNS)
    set(OPENMP_FOUND FALSE)
  endif()
endif()

#------------------------------------------------------------------------------
# Run tests to find required packages

# Check for Boost
set(BOOST_ROOT $ENV{BOOST_DIR})
set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0 1.44 1.44.0 1.45 1.45.0 1.46 1.46.0 1.46.1 1.47 1.47.0)

if (DOLFIN_ENABLE_MPI AND MPI_FOUND)
  find_package(Boost 1.36 COMPONENTS filesystem program_options system thread iostreams math_tr1 mpi serialization REQUIRED)
else()
  find_package(Boost 1.36 COMPONENTS filesystem program_options system thread iostreams math_tr1 REQUIRED)
endif()

# Check for required packages
find_package(UFC 2.0.2 QUIET HINTS ${UFC_DIR})
if (NOT UFC_FOUND)
  message(FATAL_ERROR "Could not find a configuration file for package UFC that is "
    "compatible with requested version 2.0.2.\n"
    "Set UFC_DIR to the directory containing a CMake configuration file for UFC.")
else()
  message(STATUS "UFC version: ${UFC_VERSION_STRING}")
endif()

find_package(Armadillo REQUIRED)
find_package(LibXml2 REQUIRED)

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Note: Check for Python interpreter even when Python is disabled because it
#       is used to get the installation path for dolfin_utils
find_package(PythonInterp)
if (DOLFIN_ENABLE_PYTHON)
  find_package(PythonLibs)

  # If Python is found, check for NumPy and SWIG
  if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    find_package(NumPy REQUIRED)

    find_package(SWIG REQUIRED)
    if (${SWIG_VERSION} LESS 2.0)
      message(FATAL_ERROR " DOLFIN requires SWIG version 2.0 or greater. You have version ${SWIG_VERSION}. Set DOLFIN_ENABLE_PYTHON to False or install correct SWIG version.")
    endif()
    include(UseSWIG)
    set(PYTHON_FOUND TRUE)
  endif()

endif()

# Check for ParMETIS and SCOTCH
if (DOLFIN_ENABLE_MPI AND MPI_FOUND)
  if (DOLFIN_ENABLE_PARMETIS)
    find_package(ParMETIS)
  endif()

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

# Check for PaStiX
if (DOLFIN_ENABLE_PASTIX)
  find_package(PaStiX)
endif()

# Check for Trilinos and the requires Trilinos packages
if (DOLFIN_ENABLE_TRILINOS)
  message(STATUS "Checking for Trilinos")
  find_package(Trilinos 10.10.0 PATHS ${TRILINOS_DIR} ${Trilinos_DIR} $ENV{TRILINOS_DIR} QUIET)
  set(DOLFIN_TRILINOS_PACKAGES "Epetra;Zoltan;ML;Ifpack;Amesos")

  # Check for required packages
  set(DOLFIN_TRILINOS_PACKAGES_FOUND true)
  if (Trilinos_FOUND)
    message(STATUS "  Trilinos version ${Trilinos_VERSION} found. Checking for components")

    # Check that necessary packages are enabled
    set(DOLFIN_TRILINOS_PACKAGES_FOUND true)
    foreach (required_package ${DOLFIN_TRILINOS_PACKAGES})

      # Search for required package in list of available packages
      set(${required_trilinos_package}_found false)
      list(FIND Trilinos_PACKAGE_LIST ${required_package} required_trilinos_package_found)

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

# Check for CGAL
if (DOLFIN_ENABLE_CGAL)
  find_package(CGAL 3.8)
endif()

# Check for zlib
if (DOLFIN_ENABLE_ZLIB)
  find_package(ZLIB)
endif()

# Check for cppunit
if (DOLFIN_ENABLE_UNIT_TESTS)
  find_package(CppUnit)
endif()

# Check for Sphinx
if (DOLFIN_ENABLE_DOCS AND PYTHON_FOUND)
  find_package(Sphinx 1.0.7)
endif()

# Check for VTK
if (DOLFIN_ENABLE_VTK)
  find_package(VTK HINTS ${VTK_DIR} $ENV{VTK_DIR})
endif()

#------------------------------------------------------------------------------
# Print summary of found and not found optional packages

# FIXME: Use FeatureSummary.cmake to do this

# Gather information about which optional packages were found and not found
set(OPTIONAL_PACKAGES_FOUND "")
set(OPTIONAL_PACKAGES_NOT_FOUND "")
set(OPTIONAL_PACKAGES_NOT_ENABLED "")
foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES})
  string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
  if (${PKG}_FOUND OR ${OPTIONAL_PACKAGE}_FOUND)
    list(APPEND OPTIONAL_PACKAGES_FOUND ${PKG})
  elseif (DOLFIN_ENABLE_${PKG})
    list(APPEND OPTIONAL_PACKAGES_NOT_FOUND ${PKG})
  else()
    list(APPEND OPTIONAL_PACKAGES_NOT_ENABLED ${PKG})
  endif()
endforeach()

message(STATUS "")

# Print information about packages that were found
if (OPTIONAL_PACKAGES_FOUND)
  message(STATUS "The following optional packages were found:")
  message(STATUS "-------------------------------------------")
  foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES_FOUND})
    string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
    message(STATUS "(OK) ${OPTIONAL_PACKAGE}")
  endforeach()
  message(STATUS "")
endif()

# Print information about packages that were not enabled
if (OPTIONAL_PACKAGES_NOT_ENABLED)
  message(STATUS "The following optional packages were not enabled:")
  message(STATUS "-------------------------------------------------")
  foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES_NOT_ENABLED})
    string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
    message(STATUS "(--) ${OPTIONAL_PACKAGE}")
  endforeach()
  message(STATUS "")
endif()

# Print information about packages that were not found
if (OPTIONAL_PACKAGES_NOT_FOUND)
  message(STATUS "The following optional packages could not be found:")
  message(STATUS "---------------------------------------------------")
  foreach (OPTIONAL_PACKAGE ${OPTIONAL_PACKAGES_NOT_FOUND})
    string(TOUPPER "${OPTIONAL_PACKAGE}" PKG)
    message(STATUS "(**) ${OPTIONAL_PACKAGE}")
  endforeach()
  message(STATUS "")
endif()

#------------------------------------------------------------------------------
# Get installation paths for Python modules (pure and platform-dependent)

if (PYTHONINTERP_FOUND)

  if (NOT DEFINED DOLFIN_INSTALL_PYTHON_MODULE_DIR)
    # Get path for platform-dependent Python modules (since we install a binary libary)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}'))"
      OUTPUT_VARIABLE DOLFIN_INSTALL_PYTHON_MODULE_DIR
      )
    # Strip off CMAKE_INSTALL_PREFIX (is added later by CMake)
    string(REGEX REPLACE "${CMAKE_INSTALL_PREFIX}(/|\\\\)([^ ]*)" "\\2"
      DOLFIN_INSTALL_PYTHON_MODULE_DIR "${DOLFIN_INSTALL_PYTHON_MODULE_DIR}")
    set(DOLFIN_INSTALL_PYTHON_MODULE_DIR ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
      CACHE PATH "Python extension module installation directory.")
  endif()

  if (NOT DEFINED DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR)
    # Get path for pure Python modules
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=False, prefix='${CMAKE_INSTALL_PREFIX}'))"
      OUTPUT_VARIABLE DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR
      )
    # Strip off CMAKE_INSTALL_PREFIX (is added later by CMake)
    string(REGEX REPLACE "${CMAKE_INSTALL_PREFIX}(/|\\\\)([^ ]*)" "\\2"
      DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR "${DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR}")
    set(DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR ${DOLFIN_INSTALL_PYTHON_PURE_MODULE_DIR}
      CACHE PATH "Python module installation directory.")
  endif()

endif()

#------------------------------------------------------------------------------
# Installation of dolfin Python module

if (DOLFIN_ENABLE_PYTHON AND PYTHON_FOUND)
  install(DIRECTORY
    ${CMAKE_SOURCE_DIR}/site-packages/dolfin
    DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
    USE_SOURCE_PERMISSIONS
    COMPONENT RuntimeLibraries
    PATTERN "*.in" EXCLUDE
    )

  get_filename_component(SWIG_BINARY ${SWIG_EXECUTABLE} NAME)

  configure_file(${CMAKE_SOURCE_DIR}/site-packages/dolfin/common/globalparameters.py.in
    ${CMAKE_BINARY_DIR}/globalparameters.py @ONLY)

  install(FILES ${CMAKE_BINARY_DIR}/globalparameters.py
    DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin/common/
    COMPONENT RuntimeLibraries
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

install(DIRECTORY ${DOLFIN_SOURCE_DIR}/doc/man/
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
	PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
        COMPONENT RuntimeExecutables)

#------------------------------------------------------------------------------
# Add demos and install demo source files and mesh files

# Add demo but do not add to default target
add_subdirectory(demo EXCLUDE_FROM_ALL)

# Set make program
if ("${CMAKE_GENERATOR}" STREQUAL "Unix Makefiles")
  set(MAKE_PROGRAM "$(MAKE)")
else()
  set(MAKE_PROGRAM "${CMAKE_MAKE_PROGRAM}")
endif()

# Add target "demo" for building the demos
add_custom_target(demo
  COMMAND ${MAKE_PROGRAM}
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
    message(WARNING "Disabling generation of documentation because Sphinx is missing.")
  else()
    add_subdirectory(doc)
  endif()
endif()

#------------------------------------------------------------------------------
# Add tests and benchmarks

if (DOLFIN_ENABLE_BENCHMARKS)
  # Add bench but do not add to default target
  add_subdirectory(bench EXCLUDE_FROM_ALL)

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

if (CPPUNIT_FOUND)
  # Add test but do not add to default target
  add_subdirectory(test EXCLUDE_FROM_ALL)

  # Add target "test" for building tests
  add_custom_target(test
    COMMAND ${MAKE_PROGRAM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test")
endif()

if (DOLFIN_ENABLE_TESTING)
  # Copy files needed to run tests in build directory
  file(COPY data demo test DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.cpp"
    PATTERN "*.ufl"
    PATTERN "*.xml*"
    PATTERN "*.off"
    PATTERN "*.inp"
    PATTERN "*.msh"
    PATTERN "*.supp"
    PATTERN "*.rst"
    PATTERN "CMakeFiles" EXCLUDE)

  # Add target "run_memorytests" for running memory tests
  add_custom_target(run_memorytests
    COMMAND ${PYTHON_EXECUTABLE} "test.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/memory")

  # Add target "run_regressiontests" for running regression tests
  add_custom_target(run_regressiontests
    COMMAND ${PYTHON_EXECUTABLE} "test.py"
    DEPENDS demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/regression")

  # Add target "run_systemtests" for running system tests
  add_custom_target(run_systemtests
    COMMAND ${PYTHON_EXECUTABLE} "test.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/system")

  # Add target "run_unittests" for running unit tests
  add_custom_target(run_unittests
    COMMAND ${PYTHON_EXECUTABLE} "test.py"
    DEPENDS test
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/unit")

  # Add target "run_quicktest" for running only Python unit tests
  add_custom_target(run_quicktest
    COMMAND ${PYTHON_EXECUTABLE} test.py --only-python
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/unit")

  # Add target "run_doctest" for running documentation tests
  add_custom_target(run_doctest
    COMMAND ${PYTHON_EXECUTABLE} test.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test/documentation")

  # Add target "runtests" for running all tests
  add_custom_target(runtests
    ${PYTHON_EXECUTABLE} "test.py"
    DEPENDS demo test
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test")
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
