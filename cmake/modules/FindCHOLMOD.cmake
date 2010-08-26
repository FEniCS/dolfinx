# - Try to find CHOLMOD
# Once done this will define
#
#  CHOLMOD_FOUND        - system has CHOLMOD
#  CHOLMOD_INCLUDE_DIRS - include directories for CHOLMOD
#  CHOLMOD_LIBRARIES    - libraries for CHOLMOD

message(STATUS "Checking for package 'CHOLMOD'")

# Find packages that CHOLMOD depends on
find_package(AMD)
find_package(BLAS)
find_package(ParMETIS)

# FIXME: Should we have separate FindXX modules for CAMD, COLAMD, and CCOLAMD?
# FIXME: find_package(CAMD)
# FIXME: find_package(COLAMD)
# FIXME: find_package(CCOLAMD)

# FIXME: It may be necessary to link to LAPACK and BLAS (or the vecLib
# FIXME: framework on Darwin).

# Check for header file
find_path(CHOLMOD_INCLUDE_DIRS cholmod.h
  PATHS ${CHOLMOD_DIR}/include $ENV{CHOLMOD_DIR}/include
  PATH_SUFFIXES suitesparse ufsparse
  DOC "Directory where the CHOLMOD header is located"
 )

# Check for CHOLMOD library
find_library(CHOLMOD_LIBRARY cholmod
  PATHS ${CHOLMOD_DIR} $ENV{CHOLMOD_DIR}
  PATH_SUFFIXES lib
  DOC "The CHOLMOD library"
  )

# Check for CAMD library
find_library(CAMD_LIBRARY camd
  PATHS ${CHOLMOD_DIR} ${CAMD_DIR} $ENV{CHOLMOD_DIR} $ENV{CAMD_DIR}
  PATH_SUFFIXES lib
  DOC "The CAMD library"
  )

# Check for COLAMD library
find_library(COLAMD_LIBRARY colamd
  PATHS ${CHOLMOD_DIR} ${COLAMD_DIR} $ENV{CHOLMOD_DIR} $ENV{COLAMD_DIR}
  PATH_SUFFIXES lib
  DOC "The COLAMD library"
  )

# Check for CCOLAMD library
find_library(CCOLAMD_LIBRARY ccolamd
  PATHS ${CHOLMOD_DIR} ${CCOLAMD_DIR} $ENV{CHOLMOD_DIR} $ENV{CCOLAMD_DIR}
  PATH_SUFFIXES lib
  DOC "The CCOLAMD library"
  )

# Check for LAPACK library
find_package(LAPACK)

# Collect libraries
set(CHOLMOD_LIBRARIES "${CHOLMOD_LIBRARY};${AMD_LIBRARIES};${CAMD_LIBRARY};${COLAMD_LIBRARY};${CCOLAMD_LIBRARY};${BLAS_LIBRARIES};${PARMETIS_LIBRARIES};${LAPACK_LIBRARIES}")

mark_as_advanced(
  CHOLMOD_INCLUDE_DIRS
  CHOLMOD_LIBRARY
  CHOLMOD_LIBRARIES
  CAMD_LIBRARY
  COLAMD_LIBRARY
  CCOLAMD_LIBRARY
  )

# Try to run a test program that uses CHOLMOD
if (CHOLMOD_INCLUDE_DIRS AND CHOLMOD_LIBRARIES)

  set(CMAKE_REQUIRED_INCLUDES "${CHOLMOD_INCLUDE_DIRS};${AMD_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_LIBRARIES ${CHOLMOD_LIBRARIES})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
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
}
" CHOLMOD_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
  "CHOLMOD could not be found. Be sure to set CHOLMOD_DIR."
  CHOLMOD_INCLUDE_DIRS CHOLMOD_LIBRARIES CHOLMOD_TEST_RUNS)

# FIXME: Use in all tests?
find_package_message(CHOLMOD "Found CHOLMOD: ${CHOLMOD_LIBRARIES}"
  "[${CHOLMOD_LIBRARIES}][${CHOLMOD_INCLUDE_DIRS}]")
