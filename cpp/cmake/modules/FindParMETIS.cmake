#=============================================================================
# - Try to find ParMETIS
#
# Once done this will define:
#
#  ParMETIS_FOUND   - system has ParMETIS
#  ParMETIS_VERSION - version of ParMETIS
#
# and the imported targets:
#
#  ParMETIS::ParMETIS
#  GKLib::GKLib (if found)
#  METIS::METIS (if found)
#
#=============================================================================
# Copyright (C) 2026 Garth N. Wells, Anders Logg, Johannes Ring, Jack S. Hale
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

include(FindPackageHandleStandardArgs)
include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

find_package(MPI 3 REQUIRED COMPONENTS CXX)

find_library(PARMETIS_LIBRARY parmetis DOC "Path to the ParMETIS library.")

find_path(
  PARMETIS_INCLUDE_DIR
  parmetis.h
  DOC "Directory where the ParMETIS header files are located."
)

find_library(METIS_LIBRARY metis DOC "Path to the METIS library.")

# Newer METIS and ParMETIS build against separate GKLib
find_library(GKLIB_LIBRARY gklib DOC "Path to the gklib library.")

# Build the list of link libraries for the compile/link test
set(_parmetis_link_libraries ${PARMETIS_LIBRARY})
if(METIS_LIBRARY)
  list(APPEND _parmetis_link_libraries ${METIS_LIBRARY})
endif()
if(GKLIB_LIBRARY)
  list(APPEND _parmetis_link_libraries ${GKLIB_LIBRARY})
endif()

# Identify the ParMETIS version from the header. Read rather than compiled
# and run, so that it also works when cross-compiling.
if(PARMETIS_INCLUDE_DIR)
  set(ParMETIS_VERSION "")
  foreach(part MAJOR MINOR SUBMINOR)
    file(
      STRINGS "${PARMETIS_INCLUDE_DIR}/parmetis.h"
      _line
      REGEX "^[ \t]*#[ \t]*define[ \t]+PARMETIS_${part}_VERSION[ \t]+[0-9]+"
    )
    if(_line MATCHES "PARMETIS_${part}_VERSION[ \t]+([0-9]+)")
      if(ParMETIS_VERSION)
        string(APPEND ParMETIS_VERSION ".${CMAKE_MATCH_1}")
      else()
        set(ParMETIS_VERSION "${CMAKE_MATCH_1}")
      endif()
    endif()
    unset(_line)
  endforeach()
  if(NOT ParMETIS_VERSION)
    message(
      WARNING
      "ParMETIS: no PARMETIS_*_VERSION in ${PARMETIS_INCLUDE_DIR}/parmetis.h."
    )
  endif()
endif()

# Build and run a functional test program
if(DOLFINX_SKIP_BUILD_TESTS)
  # skip
elseif(PARMETIS_INCLUDE_DIR AND PARMETIS_LIBRARY)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${PARMETIS_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX ${_parmetis_link_libraries})
  check_cxx_source_compiles(
    "
#define MPICH_IGNORE_CXX_SEEK 1
#include <stddef.h>
#include <mpi.h>
#include <parmetis.h>

int main()
{
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  idx_t vtxdist[2] = {0, 1}, xadj[2] = {0, 0}, adjncy[1] = {0};
  idx_t ncon = 1, nparts = 1, wgtflag = 0, numflag = 0, edgecut = 0;
  idx_t options[3] = {0, 0, 0}, part[1] = {0};
  real_t tpwgts[1] = {1.0}, ubvec[1] = {1.05};
  ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, NULL, NULL, &wgtflag, &numflag,
                       &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
  MPI_Finalize();
  return 0;
}
"
    PARMETIS_TEST_COMPILES
  )
  if(NOT PARMETIS_TEST_COMPILES AND NOT ParMETIS_FIND_QUIETLY)
    message(WARNING "ParMETIS: Simple test executable did not compile.")
  endif()
  cmake_pop_check_state()
endif()

# Standard package handling. The version comes from the header, so it is
# checked either way; DOLFINX_SKIP_BUILD_TESTS only drops the requirement
# that the functional test program builds.
set(_parmetis_required_vars PARMETIS_LIBRARY PARMETIS_INCLUDE_DIR)
if(NOT DOLFINX_SKIP_BUILD_TESTS)
  list(APPEND _parmetis_required_vars PARMETIS_TEST_COMPILES)
endif()
find_package_handle_standard_args(
  ParMETIS
  REQUIRED_VARS ${_parmetis_required_vars}
  VERSION_VAR ParMETIS_VERSION
  HANDLE_VERSION_RANGE
  FAIL_MESSAGE "ParMETIS could not be found/configured."
)
unset(_parmetis_required_vars)

if(ParMETIS_FOUND AND NOT TARGET ParMETIS::ParMETIS)
  if(METIS_LIBRARY AND NOT TARGET METIS::METIS)
    add_library(METIS::METIS UNKNOWN IMPORTED)
    set_target_properties(
      METIS::METIS
      PROPERTIES IMPORTED_LOCATION "${METIS_LIBRARY}"
    )
  endif()
  if(GKLIB_LIBRARY AND NOT TARGET GKLib::GKLib)
    add_library(GKLib::GKLib UNKNOWN IMPORTED)
    set_target_properties(
      GKLib::GKLib
      PROPERTIES IMPORTED_LOCATION "${GKLIB_LIBRARY}"
    )
  endif()

  add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
  set_target_properties(
    ParMETIS::ParMETIS
    PROPERTIES
      IMPORTED_LOCATION "${PARMETIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${PARMETIS_INCLUDE_DIR}"
  )
  target_link_libraries(
    ParMETIS::ParMETIS
    INTERFACE
      MPI::MPI_CXX
      $<$<BOOL:${METIS_LIBRARY}>:METIS::METIS>
      $<$<BOOL:${GKLIB_LIBRARY}>:GKLib::GKLib>
  )
endif()

mark_as_advanced(
  PARMETIS_LIBRARY
  PARMETIS_INCLUDE_DIR
  METIS_LIBRARY
  GKLIB_LIBRARY
)
