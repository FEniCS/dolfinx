# - Try to find UMFPACK
# Once done this will define
#
#  UMFPACK_FOUND        - system has UMFPACK
#  UMFPACK_INCLUDE_DIRS - include directories for UMFPACK
#  UMFPACK_LIBRARIES    - libraries for UMFPACK

message(STATUS "Checking for package 'UMFPACK'")

# Find packages that UMFPACK depends on
#set(CMAKE_LIBRARY_PATH ${BLAS_DIR}/lib $ENV{BLAS_DIR}/lib ${CMAKE_LIBRARY_PATH})
find_package(AMD QUIET)
find_package(BLAS QUIET)
find_package(CHOLMOD QUIET)

# Check for header file
find_path(UMFPACK_INCLUDE_DIRS umfpack.h
  HINTS ${UMFPACK_DIR}/include $ENV{UMFPACK_DIR}/include ${PETSC_INCLUDE_DIRS} $ENV{PETSC_DIR}/include
  PATH_SUFFIXES suitesparse ufsparse
  DOC "Directory where the UMFPACK header is located")
mark_as_advanced(UMFPACK_INCLUDE_DIRS)

# Check for UMFPACK library
find_library(UMFPACK_LIBRARY umfpack
  HINTS ${UMFPACK_DIR}/lib $ENV{UMFPACK_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The UMFPACK library")
mark_as_advanced(UMFPACK_LIBRARY)

#  Check for SUITESPARSE library on Apple
if (APPLE)
  find_library(SUITESPARSE_LIBRARY SuiteSparse
    HINTS ${UMFPACK_DIR}/lib $ENV{UMFPACK_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
    DOC "The SUITESPARSE library")
  mark_as_advanced(SUITESPARSE_LIBRARY)
endif()

# Check for SUITESPARSECONFIG library
find_library(SUITESPARSECONFIG_LIBRARY suitesparseconfig
  HINTS ${UMFPACK_DIR}/lib $ENV{UMFPACK_DIR}/lib ${PETSC_LIBRARY_DIRS} $ENV{PETSC_DIR}/lib
  DOC "The SUITESPARSE library")
mark_as_advanced(SUITESPARSECONFIG_LIBRARY)

# Collect libraries
if (AMD_FOUND)
  set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARY} ${AMD_LIBRARIES})
endif()
if (BLAS_FOUND)
  set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()
if (CHOLMOD_FOUND)
  set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${CHOLMOD_LIBRARIES})
endif()

if (SUITESPARSE_LIBRARY)
  set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${SUITESPARSE_LIBRARY})
endif()
if (SUITESPARSECONFIG_LIBRARY)
  set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${SUITESPARSECONFIG_LIBRARY})
endif()

find_program(GFORTRAN_EXECUTABLE gfortran)
if (GFORTRAN_EXECUTABLE)
  execute_process(COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.so
  OUTPUT_VARIABLE GFORTRAN_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (EXISTS "${GFORTRAN_LIBRARY}")
    set(UMFPACK_LIBRARIES ${UMFPACK_LIBRARIES} ${GFORTRAN_LIBRARY})
  endif()
endif()

# Try compiling and running test program
if (DOLFIN_SKIP_BUILD_TESTS)
  set(UMFPACK_TEST_RUNS TRUE)
elseif (UMFPACK_INCLUDE_DIRS AND UMFPACK_LIBRARIES AND AMD_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${UMFPACK_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${UMFPACK_LIBRARIES})

  set(UMFPACK_TEST_LIB_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/umfpack_test_lib.cpp")
  file(WRITE ${UMFPACK_TEST_LIB_CPP} "
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
")

  try_run(UMFPACK_TEST_LIB_EXITCODE
    UMFPACK_TEST_LIB_COMPILED
    ${CMAKE_CURRENT_BINARY_DIR}
    ${UMFPACK_TEST_LIB_CPP}
    CMAKE_FLAGS
    "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
    "-DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}"
    COMPILE_OUTPUT_VARIABLE UMFPACK_TEST_LIB_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE UMFPACK_TEST_LIB_OUTPUT
    )

  if (UMFPACK_TEST_LIB_COMPILED AND UMFPACK_TEST_LIB_EXITCODE EQUAL 0)
    message(STATUS "Performing test UMFPACK_TEST_RUNS - Success")
    set(UMFPACK_TEST_RUNS TRUE)
  else()
    message(STATUS "Performing test UMFPACK_TEST_RUNS - Failed")
  endif()

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
  UMFPACK_LIBRARIES UMFPACK_INCLUDE_DIRS AMD_LIBRARIES BLAS_LIBRARIES
  UMFPACK_TEST_RUNS)
