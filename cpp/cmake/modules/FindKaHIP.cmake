#=============================================================================
# - Try to find KaHIP
# Once done this will define
#
#  KaHIP_FOUND        - system has KaHIP
#  KaHIP::parhip      - imported target for parhip_interface library
#  KaHIP::kahip       - imported target for kahip library
#  KaHIP::KaHIP       - interface target linking all KaHIP components
#
#=============================================================================
# Copyright (C) 2026 Igor A. Baratta, Jack S. Hale
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

message(STATUS "Checking for package 'KaHIP'")

find_package(MPI REQUIRED COMPONENTS CXX)

find_path(KAHIP_INCLUDE_DIR parhip_interface.h PATH_SUFFIXES kahip)
find_library(PARHIP_LIBRARY parhip_interface)
find_library(KAHIP_LIBRARY kahip)
mark_as_advanced(KAHIP_INCLUDE_DIR PARHIP_LIBRARY KAHIP_LIBRARY)

include(FindPackageHandleStandardArgs)
if(DOLFINX_SKIP_BUILD_TESTS)
  find_package_handle_standard_args(
    KaHIP
    "KaHIP could not be found/configured."
    KAHIP_INCLUDE_DIR
    PARHIP_LIBRARY
  )
else()
  if(PARHIP_LIBRARY AND KAHIP_INCLUDE_DIR)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${KAHIP_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${PARHIP_LIBRARY})
    if(KAHIP_LIBRARY)
      list(APPEND CMAKE_REQUIRED_LIBRARIES ${KAHIP_LIBRARY})
    endif()
    list(APPEND CMAKE_REQUIRED_LINK_LIBRARIES MPI::MPI_CXX)
    check_cxx_source_compiles(
      "
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
        int *vwgt = nullptr;
        int *adjcwgt = nullptr;
        kaffpa(&n, vwgt, xadj.data(), adjcwgt, adjncy.data(),
               &nparts, &imbalance, false, 0, ECO, &edge_cut,
               part.data());
        return 0;
      }
      "
      KAHIP_TEST_COMPILES
    )
    cmake_pop_check_state()
  endif()
  find_package_handle_standard_args(
    KaHIP
    "KaHIP could not be found/configured."
    KAHIP_INCLUDE_DIR
    PARHIP_LIBRARY
    KAHIP_TEST_COMPILES
  )
endif()

if(KaHIP_FOUND)
  if(NOT TARGET KaHIP::parhip)
    add_library(KaHIP::parhip UNKNOWN IMPORTED)
    set_target_properties(
      KaHIP::parhip
      PROPERTIES
        IMPORTED_LOCATION "${PARHIP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KAHIP_INCLUDE_DIR}"
    )
    target_link_libraries(KaHIP::parhip INTERFACE MPI::MPI_CXX)
  endif()

  if(KAHIP_LIBRARY AND NOT TARGET KaHIP::kahip)
    add_library(KaHIP::kahip UNKNOWN IMPORTED)
    set_target_properties(
      KaHIP::kahip
      PROPERTIES
        IMPORTED_LOCATION "${KAHIP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KAHIP_INCLUDE_DIR}"
    )
  endif()

  if(NOT TARGET KaHIP::KaHIP)
    add_library(KaHIP::KaHIP INTERFACE IMPORTED)
    target_link_libraries(KaHIP::KaHIP INTERFACE KaHIP::parhip)
    if(TARGET KaHIP::kahip)
      target_link_libraries(KaHIP::KaHIP INTERFACE KaHIP::kahip)
    endif()
  endif()
endif()
