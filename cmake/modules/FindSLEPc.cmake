# - Try to find SLEPC
# Once done this will define
#
#  SLEPC_FOUND           - system has SLEPc
#  SLEPC_INCLUDE_DIRS    - include directories for SLEPc
#  SLEPC_LIBRARY_DIRS    - library directories for SLEPc
#  SLEPC_LIBARIES        - libraries for SLEPc
#  SLEPC_STATIC_LIBARIES - static libraries for SLEPc
#  SLEPC_VERSION         - version of SLEPc

#=============================================================================
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg and Johannes Ring
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

message(STATUS "Checking for package 'SLEPc'")

# Find SLEPc pkg-config file
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{SLEPC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(SLEPC SLEPc)

# Loop over SLEPc libraries and get absolute paths
set(_SLEPC_LIBRARIES)
foreach (lib ${SLEPC_LIBRARIES})
  find_library(LIB_${lib} ${lib} PATHS ${SLEPC_LIBRARY_DIRS} NO_DEFAULT_PATH)
  list(APPEND _SLEPC_LIBRARIES ${LIB_${lib}})
endforeach()

# Set libaries with absolute paths to SLEPC_LIBRARIES
set(SLEPC_LIBRARIES ${_SLEPC_LIBRARIES})

set(SLEPC_REQUIRES_STATIC_LIB FALSE)

# Compile and run test
if (DOLFIN_SKIP_BUILD_TESTS)

  set(SLEPC_TEST_RUNS TRUE)

elseif (SLEPC_LIBRARIES AND SLEPC_INCLUDE_DIRS)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${SLEPC_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${SLEPC_LIBRARIES})

  # Add MPI variables if MPI has been found
  if (MPI_C_FOUND)
    set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${MPI_C_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_C_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS     "${CMAKE_REQUIRED_FLAGS} ${MPI_C_COMPILE_FLAGS}")
  endif()

  # Run SLEPc test program
  set(SLEPC_TEST_LIB_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/slepc_test_lib.cpp")
  file(WRITE ${SLEPC_TEST_LIB_CPP} "
#include \"petsc.h\"
#include \"slepceps.h\"
int main()
{
  PetscErrorCode ierr;
  int argc = 0;
  char** argv = NULL;
  ierr = SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  EPS eps;
  ierr = EPSCreate(PETSC_COMM_SELF, &eps); CHKERRQ(ierr);
  //ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  ierr = EPSDestroy(eps); CHKERRQ(ierr);
#else
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
#endif
  ierr = SlepcFinalize(); CHKERRQ(ierr);
  return 0;
}
")

  try_run(
    SLEPC_TEST_LIB_EXITCODE
    SLEPC_TEST_LIB_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SLEPC_TEST_LIB_CPP}
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE SLEPC_TEST_LIB_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE SLEPC_TEST_LIB_OUTPUT
    )

  if (SLEPC_TEST_LIB_COMPILED AND SLEPC_TEST_LIB_EXITCODE EQUAL 0)
    message(STATUS "Performing test SLEPC_TEST_RUNS with shared library linking - Success")
    set(SLEPC_TEST_RUNS TRUE)
  else()
    message(STATUS "Performing test SLEPC_TEST_RUNS with shared library linking - Failed")

    # Loop over SLEPcstatic libraries and get absolute paths
    set(_SLEPC_STATIC_LIBRARIES)
    foreach (lib ${SLEPC_STATIC_LIBRARIES})
      find_library(LIB_${lib} ${lib} HINTS ${SLEPC_STATIC_LIBRARY_DIRS})
      list(APPEND _SLEPC_STATIC_LIBRARIES ${LIB_${lib}})
    endforeach()

    # Copy libaries with  absolute paths to PETSC_LIBRARIES
    set(SLEPC_STATIC_LIBRARIES ${_SLEPC_STATIC_LIBRARIES})

    # Set flags for building test program (static libs)
    set(CMAKE_REQUIRED_INCLUDES ${SLEPC_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${SLEPC_STATIC_LIBRARIES})

    try_run(
      SLEPC_TEST_STATIC_LIBS_EXITCODE
      SLEPC_TEST_STATIC_LIBS_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${SLEPC_TEST_LIB_CPP}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
        "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE SLEPC_TEST_STATIC_LIBS_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE SLEPC_TEST_STATIC_LIBS_OUTPUT
      )

    if (SLEPC_TEST_STATIC_LIBS_COMPILED AND SLEPC_STATIC_LIBS_EXITCODE EQUAL 0)
      message(STATUS "Performing test SLEPC_TEST__RUNS with static linking - Success")
      set(SLEPC_TEST_RUNS TRUE)
      set(SLEPC_REQUIRES_STATIC_LIB TRUE)
    else()
      message(STATUS "Performing test SLEPC_TETS_RUNS with static lining - Failed")
      set(SLEPC_TEST_RUNS FALSE)
    endif()
  endif()
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPc
  "SLEPc could not be found. Be sure to set SLEPC_DIR.."
  SLEPC_LIBRARY_DIRS SLEPC_LIBRARIES SLEPC_REQUIRES_STATIC_LIB
  SLEPC_INCLUDE_DIRS SLEPC_TEST_RUNS SLEPC_VERSION)
