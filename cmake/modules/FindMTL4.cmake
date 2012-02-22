# - Try to find MTL4
# Once done this will define
#
#  MTL4_FOUND        - system has MTL4
#  MTL4_INCLUDE_DIRS - include directories for MTL4
#  MTL4_LIBRARIES    - libaries defintions for MTL4
#  MTL4_DEFINITIONS  - compiler defintions for MTL4

#=============================================================================
# Copyright (C) 2010-2011 Garth N. Wells, Anders Logg and Johannes Ring
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

message(STATUS "Checking for package 'MTL4'")

# Check for header file
find_path(MTL4_INCLUDE_DIRS boost/numeric/mtl/mtl.hpp
  HINTS ${MTL4_DIR} $ENV{MTL4_DIR}
  PATH_SUFFIXES include
  DOC "Directory where the MTL4 header is located"
  )

# Check for BLAS and enable if found
find_package(BLAS QUIET)
if (BLAS_FOUND)
  set(MTL4_LIBRARIES ${BLAS_LIBRARIES})
  set(MTL4_DEFINITIONS "-DMTL_HAS_BLAS")
endif()

# Try compiling and running test program
if (MTL4_INCLUDE_DIRS)

  # Find Boost, needed by MTL4
  set(BOOST_ROOT $ENV{BOOST_DIR})
  set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0)
  find_package(Boost REQUIRED)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES  ${MTL4_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${MTL4_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${MTL4_DEFINITIONS})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>
int main()
{
  mtl::dense_vector<double> x(10);
  int size = mtl::num_rows(x);
  return 0;
}
" MTL4_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MTL4
  "MTL4 could not be found. Be sure to set MTL4_DIR"
  MTL4_INCLUDE_DIRS MTL4_TEST_RUNS)
