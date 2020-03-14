# - Try to find PETSc
# Once done this will define
#
#  PETSC_FOUND             - system has PETSc
#  PETSC_INCLUDE_DIRS      - include directories for PETSc
#  PETSC_LIBRARY_DIRS      - library directories for PETSc
#  PETSC_LIBRARIES         - libraries for PETSc
#  PETSC_STATIC_LIBRARIES  - libraries for PETSc (static linking,
#                            undefined if not required)
#  PETSC_VERSION           - version for PETSc
#  PETSC_VERSION_MAJOR     - First number in PETSC_VERSION
#  PETSC_VERSION_MINOR     - Second number in PETSC_VERSION
#  PETSC_VERSION_SUBMINOR  - Third number in PETSC_VERSION
#  PETSC_INT_SIZE          - sizeof(PetscInt)
#  PETSC_SCALAR_COMPLEX    - PETSc is complied with complex scalar type
#
# Variables used by this module, they can change the default behaviour and
# need to be set before calling find_package:
#
#  PETSC_DEBUG             - Set this to TRUE to enable debugging output
#                            of FindPETSc.cmake if you are having problems.
#                            Please enable this before filing any bug reports.

#=============================================================================
# Copyright (C) 2010-2019 Garth N. Wells, Anders Logg and Johannes Ring
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

# Outline:
# 1. Get flags from PETSc-generated pkg-config file
# 2. Test compile and run program using shared library linking
# 3. If shared library linking fails, test with static library linking

# Load pkg-config module (provided by CMake)
find_package(PkgConfig REQUIRED)

# Find PETSc pkg-config file. Note: craypetsc_real is on Cray systems
set(ENV{PKG_CONFIG_PATH} "$ENV{CRAY_PETSC_PREFIX_DIR}/lib/pkgconfig:$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PETSC craypetsc_real PETSc)

# Extract major, minor, etc from version string
if (PETSC_VERSION)
  string(REPLACE "." ";" VERSION_LIST ${PETSC_VERSION})
  list(GET VERSION_LIST 0 PETSC_VERSION_MAJOR)
  list(GET VERSION_LIST 1 PETSC_VERSION_MINOR)
  list(GET VERSION_LIST 2 PETSC_VERSION_SUBMINOR)
endif()

# Configure PETSc IMPORT (this involves creating an 'imported' target
# and attaching 'properties')
if (PETSC_FOUND AND NOT TARGET PETSC::petsc)
  add_library(PETSC::petsc INTERFACE IMPORTED)

  # Add include paths
  set_property(TARGET PETSC::petsc PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${PETSC_INCLUDE_DIRS})

  # Add libraries
  unset(_libs)
  foreach (lib ${PETSC_LIBRARIES})
    find_library(LIB_${lib} NAMES ${lib} PATHS ${PETSC_LIBRARY_DIRS} NO_DEFAULT_PATH)
    list(APPEND _libs ${LIB_${lib}})
  endforeach()
  set_property(TARGET PETSC::petsc PROPERTY INTERFACE_LINK_LIBRARIES "${_libs}")
endif()

# Configure PETSc 'static' IMPORT (this involves creating an
# 'imported' target and attaching 'properties')
if (PETSC_FOUND AND NOT TARGET PETSC::petsc_static)
  add_library(PETSC::petsc_static INTERFACE IMPORTED)

  # Add libraries (static)
  unset(_libs)
  foreach (lib ${PETSC_STATIC_LIBRARIES})
    find_library(LIB_${lib} ${lib} HINTS ${PETSC_STATIC_LIBRARY_DIRS})
    list(APPEND _libs ${LIB_${lib}})
  endforeach()
  set_property(TARGET PETSC::petsc_static PROPERTY INTERFACE_LINK_LIBRARIES "${_libs}")
endif()

# Attempt to build and run PETSc test program
if (DOLFINX_SKIP_BUILD_TESTS)

  # Assume PETSc works
  set(PETSC_TEST_RUNS TRUE)

elseif (PETSC_FOUND)

  # Create PETSc test program
  set(PETSC_TEST_LIB_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/petsc_test_lib.cpp")
  file(WRITE ${PETSC_TEST_LIB_CPP} "
#include \"petscts.h\"
#include \"petsc.h\"
int main()
{
  PetscErrorCode ierr;
  TS ts;
  int argc = 0;
  char** argv = NULL;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
")

  # Add MPI variables if MPI has been found
  if (MPI_C_FOUND)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${MPI_C_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_C_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MPI_C_COMPILE_FLAGS}")
    set(MPI_LINK_LIBRARIES MPI::MPI_C)
  endif()

  # Try to run test program (shared linking)
  try_run(
    PETSC_TEST_LIB_EXITCODE
    PETSC_TEST_LIB_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PETSC_TEST_LIB_CPP}
    CMAKE_FLAGS
    "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
    "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    LINK_LIBRARIES PETSC::petsc "${MPI_LINK_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE PETSC_TEST_LIB_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE PETSC_TEST_LIB_OUTPUT)

  # Check program output
  if (PETSC_TEST_LIB_COMPILED AND PETSC_TEST_LIB_EXITCODE EQUAL 0)

    message(STATUS "Test PETSC_TEST_RUNS with shared library linking - Success")
    set(PETSC_TEST_RUNS TRUE)

    # Static libraries not required, so unset
    set_property(TARGET PETSC::petsc_static PROPERTY INTERFACE_LINK_LIBRARIES)

  else()

    message(STATUS "Test PETSC_TEST_RUNS with shared library linking - Failed")

    if (PETSC_DEBUG)
      message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                     "PETSC_TEST_LIB_COMPILED = ${PETSC_TEST_LIB_COMPILED}")
      message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                     "PETSC_TEST_LIB_COMPILE_OUTPUT = ${PETSC_TEST_LIB_COMPILE_OUTPUT}")
      message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                     "PETSC_TEST_LIB_EXITCODE = ${PETSC_TEST_LIB_EXITCODE}")
      message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                     "PETSC_TEST_LIB_OUTPUT = ${PETSC_TEST_LIB_OUTPUT}")
    endif()

    # Try to run test program (static linking)
    try_run(
      PETSC_TEST_LIB_EXITCODE
      PETSC_TEST_LIB_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${PETSC_TEST_LIB_CPP}
      CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      LINK_LIBRARIES PETSC::petsc PETSC::petsc_static "${MPI_LINK_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE PETSC_TEST_LIB_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE PETSC_TEST_LIB_OUTPUT)

    if (PETSC_TEST_LIB_COMPILED AND PETSC_TEST_LIB_EXITCODE EQUAL 0)
      message(STATUS "Test PETSC_TEST_RUNS static linking - Success")
      set(PETSC_TEST_RUNS TRUE)
    else()
      message(STATUS "Test PETSC_TEST_RUNS static linking - Failed")
      set(PETSC_TEST_RUNS FALSE)

      if (PETSC_DEBUG)
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "PETSC_TEST_LIB_COMPILED = ${PETSC_TEST_LIB_COMPILED}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "PETSC_TEST_LIB_COMPILE_OUTPUT = ${PETSC_TEST_LIB_COMPILE_OUTPUT}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "PETSC_TEST_LIB_EXITCODE = ${PETSC_TEST_LIB_EXITCODE}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "PETSC_TEST_LIB_OUTPUT = ${PETSC_TEST_LIB_OUTPUT}")
      endif()
    endif()

  endif()
endif()

# Check sizeof(PetscInt) and check scalar type
if (PETSC_INCLUDE_DIRS)
  set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${PETSC_INCLUDE_DIRS})
  set(CMAKE_EXTRA_INCLUDE_FILES petscsys.h)

  include(CheckTypeSize)
  check_type_size("PetscInt" PETSC_INT_SIZE)

  include(CheckSymbolExists)
  check_symbol_exists(PETSC_USE_COMPLEX petscsys.h PETSC_SCALAR_COMPLEX)

  unset(CMAKE_EXTRA_INCLUDE_FILES)
  unset(CMAKE_REQUIRED_INCLUDES)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
if (PETSC_FOUND)
  find_package_handle_standard_args(PETSc
    REQUIRED_VARS PETSC_FOUND PETSC_TEST_RUNS VERSION_VAR PETSC_VERSION
    FAIL_MESSAGE "PETSc could not be configured.")
else()
  find_package_handle_standard_args(PETSc
    REQUIRED_VARS PETSC_FOUND
    FAIL_MESSAGE "PETSc could not be found. Be sure to set PETSC_DIR as an environment variable.")
endif()
