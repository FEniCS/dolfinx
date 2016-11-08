# - Try to find SCOTCH
# Once done this will define
#
#  SCOTCH_FOUND        - system has found SCOTCH
#  SCOTCH_INCLUDE_DIRS - include directories for SCOTCH
#  SCOTCH_LIBARIES     - libraries for SCOTCH
#  SCOTCH_VERSION      - version for SCOTCH
#
# Variables used by this module, they can change the default behaviour and
# need to be set before calling find_package:
#
#  SCOTCH_DEBUG        - Set this to TRUE to enable debugging output
#                        of FindScotchPT.cmake if you are having problems.
#                        Please enable this before filing any bug reports.

#=============================================================================
# Copyright (C) 2010-2011 Garth N. Wells, Johannes Ring and Anders Logg
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

set(SCOTCH_FOUND FALSE)

message(STATUS "Checking for package 'SCOTCH-PT'")

# Check for header file
find_path(SCOTCH_INCLUDE_DIRS ptscotch.h
  HINTS ${SCOTCH_DIR}/include $ENV{SCOTCH_DIR}/include ${PETSC_INCLUDE_DIRS}
  PATH_SUFFIXES scotch
  DOC "Directory where the SCOTCH-PT header is located"
  )

# Check for scotch
find_library(SCOTCH_LIBRARY
  NAMES scotch
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib ${PETSC_LIBRARY_DIRS}
  NO_DEFAULT_PATH
  DOC "The SCOTCH library"
  )
find_library(SCOTCH_LIBRARY
  NAMES scotch
  DOC "The SCOTCH library"
  )

# Check for scotcherr
find_library(SCOTCHERR_LIBRARY
  NAMES scotcherr
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
  NO_DEFAULT_PATH
  DOC "The SCOTCH-ERROR library"
  )
find_library(SCOTCHERR_LIBRARY
  NAMES scotcherr
  DOC "The SCOTCH-ERROR library"
  )

# Check for ptscotch
find_library(PTSCOTCH_LIBRARY
  NAMES ptscotch
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib ${PETSC_LIBRARY_DIRS}
  NO_DEFAULT_PATH
  DOC "The PTSCOTCH library"
  )
find_library(PTSCOTCH_LIBRARY
  NAMES ptscotch
  DOC "The PTSCOTCH library"
  )

# Check for ptesmumps
find_library(PTESMUMPS_LIBRARY
  NAMES ptesmumps esmumps
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib ${PETSC_LIBRARY_DIRS}
  NO_DEFAULT_PATH
  DOC "The PTSCOTCH-ESMUMPS library"
  )
find_library(PTESMUMPS_LIBRARY
  NAMES ptesmumps esmumps
  DOC "The PTSCOTCH-ESMUMPS library"
  )

# Check for ptscotcherr
find_library(PTSCOTCHERR_LIBRARY
  NAMES ptscotcherr
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib ${PETSC_LIBRARY_DIRS}
  NO_DEFAULT_PATH
  DOC "The PTSCOTCH-ERROR library"
  )
find_library(PTSCOTCHERR_LIBRARY
  NAMES ptscotcherr
  DOC "The PTSCOTCH-ERROR library"
  )

set(SCOTCH_LIBRARIES ${PTSCOTCH_LIBRARY})
if (PTESMUMPS_LIBRARY)
  set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES}  ${PTESMUMPS_LIBRARY})
endif()
set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES} ${PTSCOTCHERR_LIBRARY})

# Basic check of SCOTCH_VERSION which does not require compilation
if (SCOTCH_INCLUDE_DIRS)
  file(STRINGS "${SCOTCH_INCLUDE_DIRS}/ptscotch.h" PTSCOTCH_H)
  string(REGEX MATCH "SCOTCH_VERSION [0-9]+" SCOTCH_VERSION "${PTSCOTCH_H}")
  string(REGEX MATCH "[0-9]+" SCOTCH_VERSION "${SCOTCH_VERSION}")
endif()

# If SCOTCH_VERSION was not found in ptscotch.h, look in scotch.h
if (SCOTCH_INCLUDE_DIRS AND NOT SCOTCH_VERSION)
  file(STRINGS "${SCOTCH_INCLUDE_DIRS}/scotch.h" SCOTCH_H)
  string(REGEX MATCH "SCOTCH_VERSION [0-9]+" SCOTCH_VERSION "${SCOTCH_H}")
  string(REGEX MATCH "[0-9]+" SCOTCH_VERSION "${SCOTCH_VERSION}")
endif()

# For SCOTCH version > 6, need to add libraries scotch and ptscotch
if (NOT "${SCOTCH_VERSION}" VERSION_LESS "6")
  set(SCOTCH_LIBRARIES ${PTSCOTCH_LIBRARY} ${SCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY})
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${SCOTCH_LIBRARY})
endif()

# Try compiling and running test program
if (DOLFIN_SKIP_BUILD_TESTS)
  message(STATUS "Found SCOTCH (version ${SCOTCH_VERSION})")
  set(SCOTCH_TEST_RUNS TRUE)
elseif (SCOTCH_INCLUDE_DIRS AND SCOTCH_LIBRARIES)
  if (SCOTCH_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of ptscotch.h: ${SCOTCH_INCLUDE_DIRS}/ptscotch.h")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of libscotch: ${SCOTCH_LIBRARY}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of libptscotch: ${PTSCOTCH_LIBRARY}")
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of libptscotcherr: ${PTSCOTCHERR_LIBRARY}")
  endif()

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${SCOTCH_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${SCOTCH_LIBRARIES})
  #set(CMAKE_REQUIRED_LIBRARIES ${SCOTCH_LIBRARY} ${SCOTCHERR_LIBRARY})

  # Add MPI variables if MPI has been found
  if (MPI_CXX_FOUND)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${MPI_CXX_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_CXX_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
  endif()

  set(SCOTCH_CONFIG_TEST_VERSION_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/scotch_config_test_version.cpp")
  file(WRITE ${SCOTCH_CONFIG_TEST_VERSION_CPP} "
#define MPICH_IGNORE_CXX_SEEK 1
#include <stdint.h>
#include <stdio.h>
#include <mpi.h>
#include <ptscotch.h>
#include <iostream>

int main() {
  std::cout << SCOTCH_VERSION << \".\"
	    << SCOTCH_RELEASE << \".\"
	    << SCOTCH_PATCHLEVEL;
  return 0;
}
")

  try_run(
    SCOTCH_CONFIG_TEST_VERSION_EXITCODE
    SCOTCH_CONFIG_TEST_VERSION_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SCOTCH_CONFIG_TEST_VERSION_CPP}
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE OUTPUT
    )

  # Set version number
  if (SCOTCH_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
    set(SCOTCH_VERSION ${OUTPUT} CACHE TYPE STRING)
    message(STATUS "Found SCOTCH (version ${SCOTCH_VERSION})")
  endif()

  # PT-SCOTCH was first introduced in SCOTCH version 5.0
  # FIXME: parallel graph partitioning features in PT-SCOTCH was first
  #        introduced in 5.1. Do we require version 5.1?
  if (NOT ${SCOTCH_VERSION} VERSION_LESS "5.0")
    set(SCOTCH_TEST_LIB_CPP
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/scotch_test_lib.cpp")
    file(WRITE ${SCOTCH_TEST_LIB_CPP} "
#define MPICH_IGNORE_CXX_SEEK 1
#include <stdint.h>
#include <stdio.h>
#include <mpi.h>
#include <ptscotch.h>
#include <iostream>
#include <cstdlib>

int main() {
  int provided;
  SCOTCH_Dgraph dgrafdat;

  MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);

  if (SCOTCH_dgraphInit(&dgrafdat, MPI_COMM_WORLD) != 0) {
    if (MPI_THREAD_MULTIPLE > provided) {
      std::cout << \"MPI implementation is not thread-safe:\" << std::endl;
      std::cout << \"SCOTCH should be compiled without SCOTCH_PTHREAD\" << std::endl;
      exit(1);
    }
    else {
      std::cout << \"libptscotch linked to libscotch or other unknown error\" << std::endl;
      exit(2);
    }
  }
  else {
    SCOTCH_dgraphExit(&dgrafdat);
  }

  MPI_Finalize();

  return 0;
}
")

    message(STATUS "Performing test SCOTCH_TEST_RUNS")
    try_run(
      SCOTCH_TEST_LIB_EXITCODE
      SCOTCH_TEST_LIB_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${SCOTCH_TEST_LIB_CPP}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
        "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE SCOTCH_TEST_LIB_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE SCOTCH_TEST_LIB_OUTPUT
      )

    if (SCOTCH_TEST_LIB_COMPILED AND SCOTCH_TEST_LIB_EXITCODE EQUAL 0)
      message(STATUS "Performing test SCOTCH_TEST_RUNS - Success")
      set(SCOTCH_TEST_RUNS TRUE)
    else()
      message(STATUS "Performing test SCOTCH_TEST_RUNS - Failed")
      if (SCOTCH_DEBUG)
        # Output some variables
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SCOTCH_TEST_LIB_COMPILED = ${SCOTCH_TEST_LIB_COMPILED}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SCOTCH_TEST_LIB_COMPILE_OUTPUT = ${SCOTCH_TEST_LIB_COMPILE_OUTPUT}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SCOTCH_TEST_LIB_EXITCODE = ${SCOTCH_TEST_LIB_EXITCODE}")
        message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                       "SCOTCH_TEST_LIB_OUTPUT = ${SCOTCH_TEST_LIB_OUTPUT}")
      endif()
    endif()

    # If program does not run, try adding zlib library and test again
    if(NOT SCOTCH_TEST_RUNS)
      if (NOT ZLIB_FOUND)
        find_package(ZLIB)
      endif()

      if (ZLIB_INCLUDE_DIRS AND ZLIB_LIBRARIES)
        set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${ZLIB_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${ZLIB_LIBRARIES})

        message(STATUS "Performing test SCOTCH_ZLIB_TEST_RUNS")
        try_run(
          SCOTCH_ZLIB_TEST_LIB_EXITCODE
          SCOTCH_ZLIB_TEST_LIB_COMPILED
          ${CMAKE_CURRENT_BINARY_DIR}
          ${SCOTCH_TEST_LIB_CPP}
          CMAKE_FLAGS
                  "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
                  "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
          COMPILE_OUTPUT_VARIABLE SCOTCH_ZLIB_TEST_LIB_COMPILE_OUTPUT
          RUN_OUTPUT_VARIABLE SCOTCH_ZLIB_TEST_LIB_OUTPUT
          )

        # Add zlib flags if required and set test run to 'true'
        if (SCOTCH_ZLIB_TEST_LIB_COMPILED AND SCOTCH_ZLIB_TEST_LIB_EXITCODE EQUAL 0)
          message(STATUS "Performing test SCOTCH_ZLIB_TEST_RUNS - Success")
          set(SCOTCH_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS})
          set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES} ${ZLIB_LIBRARIES})
          set(SCOTCH_TEST_RUNS TRUE)
        else()
          message(STATUS "Performing test SCOTCH_ZLIB_TEST_RUNS - Failed")
          if (SCOTCH_DEBUG)
            message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                           "SCOTCH_ZLIB_TEST_LIB_COMPILED = ${SCOTCH_ZLIB_TEST_LIB_COMPILED}")
            message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                           "SCOTCH_ZLIB_TEST_LIB_COMPILE_OUTPUT = ${SCOTCH_ZLIB_TEST_LIB_COMPILE_OUTPUT}")
            message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                           "SCOTCH_TEST_LIB_EXITCODE = ${SCOTCH_TEST_LIB_EXITCODE}")
            message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                           "SCOTCH_TEST_LIB_OUTPUT = ${SCOTCH_TEST_LIB_OUTPUT}")
          endif()
        endif()

      endif()
    endif()
  endif()
endif()

# Standard package handling
find_package_handle_standard_args(SCOTCH
                                  "SCOTCH could not be found. Be sure to set SCOTCH_DIR."
                                  SCOTCH_LIBRARIES
                                  SCOTCH_INCLUDE_DIRS
                                  SCOTCH_TEST_RUNS)
