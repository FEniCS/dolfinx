#=============================================================================
# - Try to find ParMETIS
#
# Once done this will define:
#
#  ParMETIS_FOUND   - system has ParMETIS
#  ParMETIS_VERSION - version of ParMETIS
#
# and the imported target:
#
#  ParMETIS::ParMETIS
#
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

if(MPI_CXX_FOUND)
  find_library(
    PARMETIS_LIBRARY
    parmetis
    DOC "Directory where the ParMETIS library is located."
  )

  find_path(
    PARMETIS_INCLUDE_DIR
    parmetis.h
    DOC "Directory where the ParMETIS header files are located."
  )

  find_library(
    METIS_LIBRARY
    metis
    DOC "Directory where the METIS library is located."
  )

  # Newer METIS and ParMETIS build against separate GKLib
  find_library(
    GKLIB_LIBRARY
    gklib
    DOC "Directory where the gklib library is located."
  )

  # Build the list of link libraries for the test and the imported target
  set(_parmetis_link_libraries ${PARMETIS_LIBRARY})
  if(METIS_LIBRARY)
    list(APPEND _parmetis_link_libraries ${METIS_LIBRARY})
  endif()
  if(GKLIB_LIBRARY)
    list(APPEND _parmetis_link_libraries ${GKLIB_LIBRARY})
  endif()

  # Try compiling and running test program
  if(DOLFINX_SKIP_BUILD_TESTS)
    set(PARMETIS_TEST_RUNS TRUE)
    set(ParMETIS_VERSION "UNKNOWN")
    set(PARMETIS_VERSION_OK TRUE)
  elseif(PARMETIS_INCLUDE_DIR AND PARMETIS_LIBRARY)

    # Set flags for building test program
    set(CMAKE_REQUIRED_INCLUDES ${PARMETIS_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
    set(
      CMAKE_REQUIRED_LIBRARIES
      ${_parmetis_link_libraries}
      ${MPI_CXX_LIBRARIES}
    )
    set(CMAKE_REQUIRED_FLAGS ${MPI_CXX_COMPILE_FLAGS})

    # Check ParMETIS version
    set(
      PARMETIS_CONFIG_TEST_VERSION_CPP
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/parmetis_config_test_version.cpp"
    )
    file(
      WRITE ${PARMETIS_CONFIG_TEST_VERSION_CPP}
      "
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
"
    )

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

    if(PARMETIS_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
      set(ParMETIS_VERSION ${PARMETIS_CONFIG_TEST_VERSION_OUTPUT})
      mark_as_advanced(ParMETIS_VERSION)
    endif()

    if(ParMETIS_FIND_VERSION)
      # Check if version found is >= required version
      if(NOT "${ParMETIS_VERSION}" VERSION_LESS "${ParMETIS_FIND_VERSION}")
        set(PARMETIS_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(PARMETIS_VERSION_OK TRUE)
    endif()
    mark_as_advanced(PARMETIS_VERSION_OK)

    # Build and run test program
    include(CheckCXXSourceRuns)
    check_cxx_source_runs(
      "
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>
#include <parmetis.h>

int main()
{
  // FIXME: Find a simple but sensible test for ParMETIS

  return 0;
}
"
      PARMETIS_TEST_RUNS
    )
  endif()
endif()

# Standard package handling
find_package_handle_standard_args(
  ParMETIS
  REQUIRED_VARS
    PARMETIS_LIBRARY
    PARMETIS_INCLUDE_DIR
    PARMETIS_TEST_RUNS
    PARMETIS_VERSION_OK
  VERSION_VAR ParMETIS_VERSION
  FAIL_MESSAGE "ParMETIS could not be found/configured."
)

if(ParMETIS_FOUND AND NOT TARGET ParMETIS::ParMETIS)
  add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
  set_target_properties(
    ParMETIS::ParMETIS
    PROPERTIES
      IMPORTED_LOCATION "${PARMETIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${PARMETIS_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${_parmetis_link_libraries}"
  )
  # MPI is a public dependency of ParMETIS
  target_link_libraries(ParMETIS::ParMETIS INTERFACE MPI::MPI_CXX)

  mark_as_advanced(
    PARMETIS_LIBRARY
    PARMETIS_INCLUDE_DIR
    METIS_LIBRARY
    GKLIB_LIBRARY
  )
endif()
