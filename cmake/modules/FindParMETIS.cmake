# - Try to find ParMETIS
# Once done this will define
#
#  PARMETIS_FOUND        - system has ParMETIS
#  PARMETIS_INCLUDE_DIRS - include directories for ParMETIS
#  PARMETIS_LIBRARIES    - libraries for ParMETIS

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

if (MPI_FOUND)
  find_path(PARMETIS_INCLUDE_DIRS parmetis.h
    HINTS ${PARMETIS_DIR}/include $ENV{PARMETIS_DIR}/include
    DOC "Directory where the ParMETIS header files are located"
  )

  find_library(PARMETIS_LIBRARY parmetis
    HINTS ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib
    DOC "Directory where the ParMETIS library is located"
  )

  find_library(METIS_LIBRARY metis
    HINTS ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib
    DOC "Directory where the METIS library is located"
  )

  set(PARMETIS_LIBRARIES ${PARMETIS_LIBRARY})
  if (METIS_LIBRARY)
    set(PARMETIS_LIBRARIES ${PARMETIS_LIBRARIES} ${METIS_LIBRARY})
  endif()

  # Try compiling and running test program
  if (PARMETIS_INCLUDE_DIRS AND PARMETIS_LIBRARY)

    # Set flags for building test program
    set(CMAKE_REQUIRED_INCLUDES ${PARMETIS_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${PARMETIS_LIBRARIES}  ${MPI_LIBRARIES})

    # Build and run test program
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>
#include <parmetis.h>

int main()
{
  // FIXME: Find a simple but sensible test for ParMETIS

  // Initialise MPI
  MPI::Init();

  // Finalize MPI
  MPI::Finalize();

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
                                  PARMETIS_INCLUDE_DIRS)
