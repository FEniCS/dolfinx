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


# Check for header file
find_path(KAHIP_INCLUDE_DIRS parhip_interface.h
  HINTS ${KAHIP_ROOT}/deploy $ENV{KAHIP_ROOT}/deploy/ /usr/local/KaHIP/deploy
  PATH_SUFFIXES kahip
  DOC "Directory where the KaHIP header is located"
  )


# find_library(KAHIP_LIBRARY kahip
#   HINTS ${KAHIP_ROOT}/deploy $ENV{KAHIP_ROOT}/deploy /usr/local/KaHIP/deploy
#   NO_DEFAULT_PATH
#   DOC "Directory where the KaHIP library is located"
# )
#
# find_library(KAHIP_LIBRARY kahip
#   DOC "Directory where the KaHIP library is located"
# )

find_library(KAHIP_LIBRARY parhip
  HINTS ${KAHIP_ROOT}/deploy $ENV{KAHIP_ROOT}/deploy /usr/local/KaHIP/deploy
  NO_DEFAULT_PATH
  DOC "Directory where the KaHIP library is located"
)


find_library(KAHIP_LIBRARY parhip
  DOC "Directory where the KaHIP library is located"
)


set(KAHIP_LIBRARIES ${KAHIP_LIBRARY})

# Set flags for building test program
set(CMAKE_REQUIRED_INCLUDES  ${KAHIP_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${KAHIP_LIBRARIES}    ${MPI_CXX_LIBRARIES})
set(CMAKE_REQUIRED_FLAGS     ${CMAKE_REQUIRED_FLAGS}  ${MPI_CXX_COMPILE_FLAGS})


# Build and run test program
include(CheckCXXSourceRuns)

check_cxx_source_runs("
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>

#include <parhip_interface.h>

int main()
{
return 0;
}
" KAHIP_TEST_RUNS)



include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KaHIP
                                  "KaHIP could not be found/configured."
                                  KAHIP_LIBRARIES
                                  KAHIP_INCLUDE_DIRS
                                  KAHIP_TEST_RUNS)
