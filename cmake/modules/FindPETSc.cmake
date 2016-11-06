# - Try to find PETSc
# Once done this will define
#
#  PETSC_FOUND        - system has PETSc
#  PETSC_INCLUDE_DIRS - include directories for PETSc
#  PETSC_LIBRARY_DIRS - include directories for PETSc
#  PETSC_LIBRARIES    - libraries for PETSc
#  PETSC_VERSION      - version for PETSc
#  PETSC_INT_SIZE     - sizeof(PetscInt)
#
#=============================================================================
# Copyright (C) 2010-2016 Garth N. Wells, Anders Logg and Johannes Ring
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

message(STATUS "Checking for package 'PETSc'")

# Find PETSc pkg-config file
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PETSC REQUIRED PETSc)

# Loop over PETSc libraries and get absolute paths
set(_PETSC_LIBRARIES)
foreach (lib ${PETSC_LIBRARIES})
  find_library(_LIB ${lib} PATHS ${PETSC_LIBRARY_DIRS} NO_DEFAULT_PATH)
  list(APPEND _PETSC_LIBRARIES ${_LIB})
endforeach()

# Copy libaries with  absolute paths to PETSC_LIBRARIES
set(PETSC_LIBRARIES ${_PETSC_LIBRARIES})

# Build PETSc test program
if (DOLFIN_SKIP_BUILD_TESTS)

  set(PETSC_TEST_RUNS TRUE)

elseif (PETSC_FOUND)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${PETSC_LIBRARIES})

  # Add MPI variables if MPI has been found
  if (MPI_C_FOUND)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${MPI_C_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_C_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MPI_C_COMPILE_FLAGS}")
  endif()

  # Run PETSc test program
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
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  ierr = TSDestroy(ts);CHKERRQ(ierr);
#else
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
#endif
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
")

  try_run(
    PETSC_TEST_LIB_EXITCODE
    PETSC_TEST_LIB_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PETSC_TEST_LIB_CPP}
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE PETSC_TEST_LIB_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE PETSC_TEST_LIB_OUTPUT
    )

  if (PETSC_TEST_LIB_COMPILED AND PETSC_TEST_LIB_EXITCODE EQUAL 0)
    message(STATUS "Performing test PETSC_TEST_RUNS - Success")
    set(PETSC_TEST_RUNS TRUE)
  else()
    message(STATUS "Performing test PETSC_TEST_RUNS - Failed")
  endif()

endif()

# Check sizeof(PetscInt)
if (PETSC_INCLUDE_DIRS)
  include(CheckTypeSize)
  set(CMAKE_EXTRA_INCLUDE_FILES petsc.h)
  check_type_size("PetscInt" PETSC_INT_SIZE)
  set(CMAKE_EXTRA_INCLUDE_FILES)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
  "PETSc could not be found. Be sure to set PETSC_DIR and PETSC_ARCH."
  PETSC_LIBRARY_DIRS PETSC_LIBRARIES PETSC_INCLUDE_DIRS PETSC_TEST_RUNS PETSC_VERSION)
