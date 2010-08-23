message(STATUS "Checking for package 'UMFPACK'")

# Check for header file
find_path(UMFPACK_INCLUDE_DIR umfpack.h
 PATHS $ENV{UMFPACK_DIR}/include
 PATH_SUFFIXES suitesparse
 DOC "Directory where the UMFPACK header is located"
 )
mark_as_advanced(UMFPACK_INCLUDE_DIR)

# Check for library
find_library(UMFPACK_LIBRARY umfpack
  PATHS $ENV{UMFPACK_DIR}/lib
  DOC "The UMFPACK library"
  )
mark_as_advanced(UMFPACK_LIBRARY)

# Try compiling and running test program
if (UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${UMFPACK_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${UMFPACK_LIBRARY})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
/* Test program umfpack-ex1.c */

#include <umfpack.h>

int main()
{
  int n = 5;
  double x[5];
  void *Symbolic, *Numeric;
  int i;

  int Ap[] = { 0, 2, 5, 9, 10, 12 };
  int Ai[] = { 0, 1, 0,  2, 4, 1,  2, 3, 4, 2, 1, 4 };
  double Ax[] = { 2, 3, 3, -1, 4, 4, -3, 1, 2, 2, 6, 1 };
  double b[] = { 8, 45, -3, 3, 19 };

  umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, NULL, NULL);
  umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, NULL, NULL);
  umfpack_di_free_symbolic(&Symbolic);

  umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, NULL, NULL);
  umfpack_di_free_numeric(&Numeric);

  return 0;
}
" UMFPACK_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UMFPACK
  "UMFPACK could not be found. Be sure to set UMFPACK_DIR."
  UMFPACK_INCLUDE_DIR UMFPACK_LIBRARY UMFPACK_TEST_RUNS)
