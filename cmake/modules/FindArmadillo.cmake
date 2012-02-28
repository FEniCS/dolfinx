# - Try to find Armadillo
# Once done this will define

#  ARMADILLO_FOUND        - system has Armadillo
#  ARMADILLO_INCLUDE_DIRS - include directories for Armadillo
#  ARMADILLO_LIBRARIES    - libraries for Armadillo
#  ARMADILLO_LINK_FLAGS   - link flags for Armadillo
#  ARMADILLO_VERSION      - the Armadillo version string (MAJOR.MINOR.PATCH)
#
# Setting these changes the behavior of the search
#
#  ARMADILLO_DIR - directory in which Armadillo resides

#=============================================================================
# Copyright (C) 2010-2011 Johannes Ring, Anders Logg and Garth N. Wells
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

message(STATUS "Checking for package 'Armadillo'")

# FIXME: Look for LAPACK libraries. Required on some platforms. BLAS too?
set(CMAKE_LIBRARY_PATH ${BLAS_DIR}/lib $ENV{BLAS_DIR}/lib ${CMAKE_LIBRARY_PATH})
set(CMAKE_LIBRARY_PATH ${LAPACK_DIR}/lib $ENV{LAPACK_DIR}/lib ${CMAKE_LIBRARY_PATH})
find_package(BLAS)
find_package(LAPACK)

find_path(ARMADILLO_INCLUDE_DIRS
  NAMES armadillo
  HINTS ${ARMADILLO_DIR}/include $ENV{ARMADILLO_DIR}/include
  DOC "Directory where the Armadillo header file is located"
  )
mark_as_advanced(ARMADILLO_INCLUDE_DIRS)

find_library(ARMADILLO_LIBRARIES
  NAMES armadillo
  HINTS ${ARMADILLO_DIR}/lib $ENV{ARMADILLO_DIR}/lib
  DOC "The Armadillo library"
  )
mark_as_advanced(ARMA

DILLO_LIBRARIES)

set(${ARMADILLO_LIBRARIES} "${ARMADILLO_LIBRARIES}")

# Special fixes for Mac
if (APPLE)

  # Link against the vecLib framework
  include(CMakeFindFrameworks)
  cmake_find_frameworks(vecLib)
  if (vecLib_FRAMEWORKS)
    set(ARMADILLO_LINK_FLAGS "-framework vecLib")
    mark_as_advanced(ARMADILLO_LINK_FLAGS)
  else()
    message(STATUS "vecLib framework not found.")
  endif()
endif()

if (ARMADILLO_INCLUDE_DIRS AND ARMADILLO_LIBRARIES)
  include(CheckCXXSourceRuns)

  # Armadillo needs the location of the Boost header files
  if (NOT Boost_FOUND)
    set(BOOST_ROOT $ENV{BOOST_DIR})
    set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0 1.44 1.44.0 1.45 1.45.0 1.46 1.46.0 1.46.1 1.47 1.47.0)
    find_package(Boost REQUIRED)
  endif()

  # These are needed for the try_run and check_cxx_source_runs commands below
  set(CMAKE_REQUIRED_INCLUDES ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${ARMADILLO_LIBRARIES})
  if (ARMADILLO_LINK_FLAGS)
    set(CMAKE_REQUIRED_LIBRARIES ${ARMADILLO_LINK_FLAGS} ${CMAKE_REQUIRED_LIBRARIES})
  endif()

  set(ARMADILLO_CONFIG_TEST_VERSION_CPP ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/check_armadillo/armadillo_config_test_version.cpp)
  file(WRITE ${ARMADILLO_CONFIG_TEST_VERSION_CPP} "
#include <armadillo>
#include <iostream>

using namespace arma;

int main() {
  std::cout << arma_version::major << \".\"
	    << arma_version::minor << \".\"
	    << arma_version::patch;
  return 0;
}
")

  try_run(
    ARMADILLO_CONFIG_TEST_VERSION_EXITCODE
    ARMADILLO_CONFIG_TEST_VERSION_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ARMADILLO_CONFIG_TEST_VERSION_CPP}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
    RUN_OUTPUT_VARIABLE OUTPUT
    )

  if (ARMADILLO_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
    set(ARMADILLO_VERSION ${OUTPUT} CACHE TYPE STRING)
    mark_as_advanced(ARMADILLO_VERSION)
  endif()

  set(armadillo_test_str "
#include <armadillo>

using namespace arma;

int main()
{
  int n = 20;
  double h = 1.0/(n-1);
  mat A(n, n);
  vec b(n);
  vec u(n);
  double beta = 0.2;
  double gamma = 1000;

  A.fill(0.0);
  b.fill(0.0);
  double x; int i;

  i = 0;
  A(i, i) = 1;
  b(i) = 0;
  for (i=1; i<n-1; i++) {
    x = (i-1)*h;
    A(i, i-1) = 1;  A(i, i) = -2;  A(i, i+1) = 1;
    b(i) = - h*h*gamma*exp(-beta*x);
  }
  i = n-1;  x = (i-1)*h;
  A(i, i-1) = 2;  A(i, i) = -2;

  u = solve(A, b);

  return 0;
}
")

  # Check that C++ program runs
  check_cxx_source_runs("${armadillo_test_str}" ARMADILLO_TEST_RUNS)

  # If program does not run, try adding LAPACK and BLAS libraries and test again
  foreach (ARMADILLO_DEP LAPACK BLAS)
    if(NOT ARMADILLO_TEST_RUNS)
      if (NOT ${ARMADILLO_DEP}_FOUND)
        find_package(${ARMADILLO_DEP})
      endif()

      if (${ARMADILLO_DEP}_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES ${${ARMADILLO_DEP}_LIBRARIES})
        check_cxx_source_runs("${armadillo_test_str}" ARMADILLO_${ARMADILLO_DEP}_TEST_RUNS)

        if (ARMADILLO_${ARMADILLO_DEP}_TEST_RUNS)
          list(APPEND ARMADILLO_LIBRARIES ${${ARMADILLO_DEP}_LIBRARIES})
          set(ARMADILLO_TEST_RUNS TRUE)
          break()
        endif()
      endif()
    endif()
  endforeach()

  # Add Lapack and BLAS
  if (LAPACK_FOUND)
    list(APPEND ARMADILLO_LIBRARIES ${LAPACK_LIBRARIES})
  endif()
  if (BLAS_FOUND)
    list(APPEND ARMADILLO_LIBRARIES ${BLAS_LIBRARIES})
  endif()

  # If program still does not run, try adding GFortran library
  if(NOT ARMADILLO_TEST_RUNS)
    find_program(GFORTRAN_EXECUTABLE gfortran)

    if (GFORTRAN_EXECUTABLE)
      execute_process(COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.so
      OUTPUT_VARIABLE GFORTRAN_LIBRARY
      OUTPUT_STRIP_TRAILING_WHITESPACE)

      if (EXISTS "${GFORTRAN_LIBRARY}")
          list(APPEND CMAKE_REQUIRED_LIBRARIES ${GFORTRAN_LIBRARY})
          check_cxx_source_runs("${armadillo_test_str}" ARMADILLO_GFORTRAN_TEST_RUNS)

          if (ARMADILLO_GFORTRAN_TEST_RUNS)
            list(APPEND ARMADILLO_LIBRARIES ${GFORTRAN_LIBRARY})
            set(ARMADILLO_TEST_RUNS TRUE)
          endif()
      endif()
    endif()
  endif()
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadillo
  "Armadillo could not be found. Be sure to set ARMADILLO_DIR."
  ARMADILLO_LIBRARIES ARMADILLO_INCLUDE_DIRS ARMADILLO_TEST_RUNS)
