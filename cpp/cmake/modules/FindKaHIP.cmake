# - Try to find KaHIP
# Once done this will define
#
#  KAHIP_FOUND        - system has KaHIP
#  KAHIP_INCLUDE_DIRS - include directories for KaHIP
#  KAHIP_LIBRARIES    - libraries for KaHIP
#  KAHIP_VERSION      - version for KaHIP
#=============================================================================
# Copyright (C) 2019 Igor A. Baratta
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

set(KAHIP_FOUND FALSE)

message(STATUS "Checking for package 'KaHIP'")

if (MPI_CXX_FOUND)
  # Check for header file
  find_path(KAHIP_INCLUDE_DIRS parhip_interface.h
    HINTS ${KAHIP_ROOT}/deploy $ENV{KAHIP_ROOT}/deploy/ /usr/local/KaHIP/deploy
    PATH_SUFFIXES kahip
    DOC "Directory where the KaHIP header files are located"
    )

  find_library(KAHIP_LIBRARY kahip
    HINTS ${KAHIP_ROOT}/deploy $ENV{KAHIP_ROOT}/deploy /usr/local/KaHIP/deploy
    NO_DEFAULT_PATH
    DOC "Directory where the KaHIP library is located"
  )

  find_library(KAHIP_LIBRARY kahip
    DOC "Directory where the KaHIP library is located"
  )

  find_library(MKAHIP_LIBRARY kahip
    HINTS ${KAHIP_ROOT}/deploy/parallel $ENV{KAHIP_ROOT}/deploy/parallel /usr/local/KaHIP/deploy/parallel
    NO_DEFAULT_PATH
    DOC "Directory where the parallel KaHIP library is located"
  )

  find_library(MKAHIP_LIBRARY kahip
    DOC "Directory where the parallel KaHIP library is located"
  )

  find_library(PARHIP_LIBRARY parhip
    HINTS ${KAHIP_ROOT}/deploy $ENV{KAHIP_ROOT}/deploy /usr/local/KaHIP/deploy
    NO_DEFAULT_PATH
    DOC "Directory where the ParHIP interface is located"
  )

  find_library(PARHIP_LIBRARY parhip
    DOC "Directory where the ParHIP interface is located"
  )


  set(KAHIP_LIBRARIES ${KAHIP_LIBRARY} ${PARHIP_LIBRARY} ${MKAHIP_LIBRARY})

  if (KAHIP_LIBRARIES)

    # Set flags for building test program
    set(CMAKE_REQUIRED_INCLUDES  ${KAHIP_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${KAHIP_LIBRARIES}    ${MPI_CXX_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS     ${CMAKE_REQUIRED_FLAGS}  ${MPI_CXX_COMPILE_FLAGS})

    # Build and run test program
    include(CheckCXXSourceRuns)

      check_cxx_source_runs("
      #define MPICH_IGNORE_CXX_SEEK 1
      #include <mpi.h>

      #include <kaHIP_interface.h>

      int main()
      {

      int n = 5;
      int *xadj = new int[6];
      xadj[0] = 0;
      xadj[1] = 2;
      xadj[2] = 5;
      xadj[3] = 7;
      xadj[4] = 9;
      xadj[5] = 12;

      int *adjncy = new int[12];
      adjncy[0] = 1;
      adjncy[1] = 4;
      adjncy[2] = 0;
      adjncy[3] = 2;
      adjncy[4] = 4;
      adjncy[5] = 1;
      adjncy[6] = 3;
      adjncy[7] = 2;
      adjncy[8] = 4;
      adjncy[9] = 0;
      adjncy[10] = 1;
      adjncy[11] = 3;

      double imbalance = 0.03;
      int *part = new int[n];
      int edge_cut = 0;
      int nparts = 2;
      int *vwgt = nullptr;;
      int *adjcwgt = nullptr;;

      kaffpa(&n, vwgt, xadj, adjcwgt, adjncy, &nparts, &imbalance, false, 0, ECO,
             &edge_cut, part);

      return 0;
      }
      " KAHIP_TEST_RUNS)
  endif()
endif()

find_package_handle_standard_args(KaHIP
                                  "KaHIP could not be found/configured."
                                  KAHIP_LIBRARIES
                                  KAHIP_INCLUDE_DIRS
                                  KAHIP_TEST_RUNS
                                )
