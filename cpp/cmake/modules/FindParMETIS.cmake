# - Try to find ParMETIS
# Once done this will define
#
#  PARMETIS_FOUND        - system has ParMETIS
#  PARMETIS_INCLUDE_DIRS - include directories for ParMETIS
#  PARMETIS_LIBRARIES    - libraries for ParMETIS
#  PARMETIS_VERSION      - version for ParMETIS

#=============================================================================
# Copyright (C) 2010 Garth N. Wells, Anders Logg and Johannes Ring
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

if (MPI_CXX_FOUND)
  find_path(PARMETIS_INCLUDE_DIRS parmetis.h
    HINTS ${PARMETIS_DIR}/include $ENV{PARMETIS_DIR}/include ${PETSC_INCLUDE_DIRS}
    DOC "Directory where the ParMETIS header files are located"
  )

  find_library(PARMETIS_LIBRARY parmetis
    HINTS ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib ${PETSC_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    DOC "Directory where the ParMETIS library is located"
  )
  find_library(PARMETIS_LIBRARY parmetis
    DOC "Directory where the ParMETIS library is located"
  )

  find_library(METIS_LIBRARY metis
    HINTS ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib ${PETSC_LIBRARY_DIRS}
    NO_DEFAULT_PATH
    DOC "Directory where the METIS library is located"
  )
  find_library(METIS_LIBRARY metis
    DOC "Directory where the METIS library is located"
  )

  set(PARMETIS_LIBRARIES ${PARMETIS_LIBRARY})
  if (METIS_LIBRARY)
    set(PARMETIS_LIBRARIES ${PARMETIS_LIBRARIES} ${METIS_LIBRARY})
  endif()

  # Try compiling and running test program
  if (DOLFIN_SKIP_BUILD_TESTS)
    set(PARMETIS_TEST_RUNS TRUE)
    set(PARMETIS_VERSION "UNKNOWN")
    set(PARMETIS_VERSION_OK TRUE)
  elseif (PARMETIS_INCLUDE_DIRS AND PARMETIS_LIBRARY)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES  ${PARMETIS_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_REQUIRED_LIBRARIES ${PARMETIS_LIBRARIES}    ${MPI_CXX_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS     ${CMAKE_REQUIRED_FLAGS}  ${MPI_CXX_COMPILE_FLAGS})

  # Check ParMETIS version
  set(PARMETIS_CONFIG_TEST_VERSION_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/parmetis_config_test_version.cpp")
  file(WRITE ${PARMETIS_CONFIG_TEST_VERSION_CPP} "
#define MPICH_IGNORE_CXX_SEEK 1
#include <iostream>
#include \"parmetis.h\"

int main() {
#ifdef PARMETIS_SUBMINOR_VERSION
  std::cout << PARMETIS_MAJOR_VERSION << \".\"
	    << PARMETIS_MINOR_VERSION << \".\"
            << PARMETIS_SUBMINOR_VERSION;
#else
  std::cout << PARMETIS_MAJOR_VERSION << \".\"
	    << PARMETIS_MINOR_VERSION;
#endif
  return 0;
}
")

    try_run(
      PARMETIS_CONFIG_TEST_VERSION_EXITCODE
      PARMETIS_CONFIG_TEST_VERSION_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${PARMETIS_CONFIG_TEST_VERSION_CPP}
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
	"-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
      COMPILE_OUTPUT_VARIABLE PARMETIS_CONFIG_TEST_VERSION_COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE PARMETIS_CONFIG_TEST_VERSION_OUTPUT
      )

    if (PARMETIS_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
      set(PARMETIS_VERSION ${PARMETIS_CONFIG_TEST_VERSION_OUTPUT} CACHE TYPE STRING)
      mark_as_advanced(PARMETIS_VERSION)
    endif()

    if (ParMETIS_FIND_VERSION)
      # Check if version found is >= required version
      if (NOT "${PARMETIS_VERSION}" VERSION_LESS "${ParMETIS_FIND_VERSION}")
	set(PARMETIS_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(PARMETIS_VERSION_OK TRUE)
    endif()
    mark_as_advanced(PARMETIS_VERSION_OK)

    # Build and run test program
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>
#include <parmetis.h>

int main()
{
  // FIXME: Find a simple but sensible test for ParMETIS

  return 0;
}
" PARMETIS_TEST_RUNS)

  endif()
endif()

# Standard package handling
find_package_handle_standard_args(ParMETIS
                                  "ParMETIS could not be found/configured."
                                  PARMETIS_LIBRARIES
                                  PARMETIS_TEST_RUNS
                                  PARMETIS_INCLUDE_DIRS
				  PARMETIS_VERSION
				  PARMETIS_VERSION_OK)
