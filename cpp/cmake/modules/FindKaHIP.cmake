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
  find_path(KAHIP_INCLUDE_DIRS parhip_interface.h
    HINTS ${KAHIP_DIR}/include $ENV{KAHIP_DIR}/include PATH_SUFFIXES kahip)
  find_library(PARHIP_LIBRARY parhip_interface HINTS ${KAHIP_DIR}/lib $ENV{KAHIP_DIR}/lib)
  find_library(KAHIP_LIBRARY kahip HINTS ${KAHIP_DIR}/lib $ENV{KAHIP_DIR}/lib)

  set(KAHIP_LIBRARIES ${PARHIP_LIBRARY} ${KAHIP_LIBRARY})

  include (FindPackageHandleStandardArgs)
  if (DOLFINX_SKIP_BUILD_TESTS)
    find_package_handle_standard_args(KaHIP
                                     "KaHIP could not be found/configured."
                                      KAHIP_INCLUDE_DIRS
                                      KAHIP_LIBRARIES)
  else()
    if (KAHIP_LIBRARIES AND KAHIP_LIBRARIES)

      # Build and run test program
      include(CheckCXXSourceRuns)

      # Set flags for building test program
      set(CMAKE_REQUIRED_INCLUDES  ${KAHIP_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
      set(CMAKE_REQUIRED_LIBRARIES ${KAHIP_LIBRARIES} ${MPI_CXX_LIBRARIES})
      set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
      check_cxx_source_runs("
        #define MPICH_IGNORE_CXX_SEEK 1
        #include <mpi.h>
        #include <vector>
        #include <kaHIP_interface.h>
        int main()
        {
          int n = 5;
          std::vector<int> xadj = {0, 2, 5, 7, 9, 12};
          std::vector<int> adjncy = {1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3};
          std::vector<int> part(n);
          double imbalance = 0.03;
          int edge_cut = 0;
          int nparts = 2;
          int *vwgt = nullptr;;
          int *adjcwgt = nullptr;;
          kaffpa(&n, vwgt, xadj.data(), adjcwgt, adjncy.data(),
                 &nparts, &imbalance, false, 0, ECO, &edge_cut,
                 part.data());
        return 0;
        }
        " KAHIP_TEST_RUNS)
    endif()
    find_package_handle_standard_args(KaHIP "KaHIP could not be found/configured."
                                      KAHIP_INCLUDE_DIRS
                                      KAHIP_LIBRARIES
                                      KAHIP_TEST_RUNS)
  endif()
endif()
