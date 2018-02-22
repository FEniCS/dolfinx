# - Try to find CHOLMOD
# Once done this will define
#
#  CHOLMOD_FOUND        - system has CHOLMOD
#  CHOLMOD_INCLUDE_DIRS - include directories for CHOLMOD
#  CHOLMOD_LIBRARIES    - libraries for CHOLMOD

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

message(STATUS "Checking for package 'CHOLMOD'")

# Find packages that CHOLMOD depends on
set(CMAKE_LIBRARY_PATH ${BLAS_DIR}/lib $ENV{BLAS_DIR}/lib ${CMAKE_LIBRARY_PATH})
set(CMAKE_LIBRARY_PATH ${LAPACK_DIR}/lib $ENV{LAPACK_DIR}/lib ${CMAKE_LIBRARY_PATH})
find_package(AMD QUIET)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)
find_package(ParMETIS 4.0.2 QUIET)

# FIXME: Should we have separate FindXX modules for CAMD, COLAMD, and CCOLAMD?
# FIXME: find_package(CAMD)
# FIXME: find_package(COLAMD)
# FIXME: find_package(CCOLAMD)

# FIXME: It may be necessary to link to LAPACK and BLAS (or the vecLib
# FIXME: framework on Darwin).

# Check for header file
find_path(CHOLMOD_INCLUDE_DIRS cholmod.h
  HINTS ${CHOLMOD_DIR}/include $ENV{CHOLMOD_DIR}/include ${PETSC_INCLUDE_DIRS} $ENV{PETSC_DIR}/include
  PATH_SUFFIXES suitesparse ufsparse
  DOC "Directory where the CHOLMOD header is located")

# Check for CHOLMOD library
find_library(CHOLMOD_LIBRARY cholmod
  HINTS ${CHOLMOD_DIR}/lib $ENV{CHOLMOD_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The CHOLMOD library")

# Check for CAMD library
find_library(CAMD_LIBRARY camd
  HINTS ${CHOLMOD_DIR}/lib ${CAMD_DIR}/lib $ENV{CHOLMOD_DIR}/lib
  $ENV{CAMD_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The CAMD library")

# Check for COLAMD library
find_library(COLAMD_LIBRARY colamd
  HINTS ${CHOLMOD_DIR}/lib ${COLAMD_DIR}/lib $ENV{CHOLMOD_DIR}/lib
  $ENV{COLAMD_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The COLAMD library"
  )

# Check for CCOLAMD library
find_library(CCOLAMD_LIBRARY ccolamd
  HINTS ${CHOLMOD_DIR}/lib ${CCOLAMD_DIR}/lib $ENV{CHOLMOD_DIR}/lib
  $ENV{CCOLAMD_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The CCOLAMD library")

# Check for SUITESPARSECONFIG library
find_library(SUITESPARSECONFIG_LIBRARY suitesparseconfig
  HINTS ${CHOLMOD_DIR}/lib ${CCOLAMD_DIR}/lib $ENV{CHOLMOD_DIR}/lib
  $ENV{CCOLAMD_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The SUITESPARSECONFIG library")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT APPLE)
  # Check for rt library
  find_library(RT_LIBRARY rt
    DOC "The RT library")
endif()

# Collect libraries (order is important)
set(CHOLMOD_LIBRARIES)
if (CHOLMOD_LIBRARY)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${CHOLMOD_LIBRARY})
endif()
if (AMD_FOUND)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES}  ${AMD_LIBRARIES})
endif()
if (CAMD_LIBRARY)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${CAMD_LIBRARY})
endif()
if (COLAMD_LIBRARY)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${COLAMD_LIBRARY})
endif()
if (CCOLAMD_LIBRARY)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${CCOLAMD_LIBRARY})
endif()
if (SUITESPARSECONFIG_LIBRARY)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${SUITESPARSECONFIG_LIBRARY})
endif()
if (RT_LIBRARY)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${RT_LIBRARY})
endif()

if (PARMETIS_FOUND)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${PARMETIS_LIBRARIES})
endif()
if (LAPACK_FOUND)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${LAPACK_LIBRARIES})
endif()
if (BLAS_FOUND)
  set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${BLAS_LIBRARIES})
endif()

find_program(GFORTRAN_EXECUTABLE gfortran)
if (GFORTRAN_EXECUTABLE)
  execute_process(COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.so
  OUTPUT_VARIABLE GFORTRAN_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (EXISTS "${GFORTRAN_LIBRARY}")
    set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARIES} ${GFORTRAN_LIBRARY})
  endif()
endif()

mark_as_advanced(CHOLMOD_INCLUDE_DIRS CHOLMOD_LIBRARY CHOLMOD_LIBRARIES
  CAMD_LIBRARY COLAMD_LIBRARY CCOLAMD_LIBRARY)

# Try to run a test program that uses CHOLMOD
if (DOLFIN_SKIP_BUILD_TESTS)
  set(CHOLMOD_TEST_RUNS TRUE)
elseif (CHOLMOD_INCLUDE_DIRS AND CHOLMOD_LIBRARIES AND AMD_FOUND)

  set(CMAKE_REQUIRED_INCLUDES  ${CHOLMOD_INCLUDE_DIRS} ${AMD_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${CHOLMOD_LIBRARIES})

  set(CHOLMOD_TEST_LIB_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/cholmod_test_lib.cpp")
  file(WRITE ${CHOLMOD_TEST_LIB_CPP} "
#include <stdio.h>
#include <cholmod.h>

int main()
{
  cholmod_dense *D;
  cholmod_sparse *S;
  cholmod_dense *x, *b, *r;
  cholmod_factor *L;
  double one[2] = {1,0}, m1[2] = {-1,0};
  double *dx;
  cholmod_common c;
  int n = 5;
  double K[5][5] = {{1.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 2.0,-1.0, 0.0, 0.0},
                    {0.0,-1.0, 2.0,-1.0, 0.0},
                    {0.0, 0.0,-1.0, 2.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 1.0}};
  cholmod_start (&c);
  D = cholmod_allocate_dense(n, n, n, CHOLMOD_REAL, &c);
  dx = (double*)D->x;
  for (int i=0; i < n; i++)
    for (int j=0; j < n; j++)
      dx[i+j*n] = K[i][j];
  S = cholmod_dense_to_sparse(D, 1, &c);
  S->stype = 1;
  cholmod_reallocate_sparse(cholmod_nnz(S, &c), S, &c);
  b = cholmod_ones(S->nrow, 1, S->xtype, &c);
  L = cholmod_analyze(S, &c);
/*
  cholmod_factorize(S, L, &c);
  x = cholmod_solve(CHOLMOD_A, L, b, &c);
  r = cholmod_copy_dense(b, &c);
  cholmod_sdmult(S, 0, m1, one, x, r, &c);
  cholmod_free_factor(&L, &c);
  cholmod_free_dense(&D, &c);
  cholmod_free_sparse(&S, &c);
  cholmod_free_dense(&r, &c);
  cholmod_free_dense(&x, &c);
  cholmod_free_dense(&b, &c);
  cholmod_finish(&c);
  return 0;
*/
}
")

  try_run(CHOLMOD_TEST_LIB_EXITCODE
    CHOLMOD_TEST_LIB_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CHOLMOD_TEST_LIB_CPP}
    CMAKE_FLAGS
    "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
    "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE CHOLMOD_TEST_LIB_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE CHOLMOD_TEST_LIB_OUTPUT
    )

  if (CHOLMOD_TEST_LIB_COMPILED AND CHOLMOD_TEST_LIB_EXITCODE EQUAL 0)
    message(STATUS "Performing test CHOLMOD_TEST_RUNS - Success")
    set(CHOLMOD_TEST_RUNS TRUE)
  else()
    message(STATUS "Performing test CHOLMOD_TEST_RUNS - Failed")
  endif()

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
  "CHOLMOD could not be found. Be sure to set CHOLMOD_DIR."
  CHOLMOD_LIBRARIES CHOLMOD_INCLUDE_DIRS CHOLMOD_TEST_RUNS)
