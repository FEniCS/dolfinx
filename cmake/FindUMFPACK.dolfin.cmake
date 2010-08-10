set(UMFPACK_FOUND 0)

message(STATUS "checking for package 'UMFPACK'")

# Check for header file
find_path(UMFPACK_INCLUDE_DIR suitesparse/umfpack.h
  $ENV{UMFPACK_DIR}
  /usr/local/include
  /usr/include
  DOC "Directory where the UMFPACK header is located"
  )

# Try compiling and running test program
if(UMFPACK_INCLUDE_DIR)
  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_INCLUDES ${UMFPACK_INCLUDE_DIR})
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

  if(NOT UMFPACK_TEST_RUNS)
    message("UMFPACK was found but a test program could not be run.")
  endif(NOT UMFPACK_TEST_RUNS)

endif(UMFPACK_INCLUDE_DIR)

# Report results of tests
if(UMFPACK_TEST_RUNS)
  message(STATUS "  found package 'UMFPACK'")
  set(UMFPACK_FOUND 1)
  include_directories(${UMFPACK_INCLUDE_DIR})
  add_definitions(-DHAS_UMFPACK)
else(UMFPACK_TEST_RUNS)
  message(STATUS "  package 'UMFPACK' could not be configured.")
endif(UMFPACK_TEST_RUNS)
