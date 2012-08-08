# - Try to find PaStiX
# Once done this will define
#
#  PASTIX_FOUND        - system has PaStiX
#  PASTIX_INCLUDE_DIRS - include directories for PaStiX
#  PASTIX_LIBRARIES    - libraries for PaStiX

# Check for PaStiX header file
find_path(PASTIX_INCLUDE_DIRS pastix.h
  HINTS ${PASTIX_DIR} $ENV{PASTIX_DIR} ${PASTIX_DIR}/include $ENV{PASTIX_DIR}/include
  PATH_SUFFIXES install
  DOC "Directory where the PaStiX header is located"
 )

# Check for PaStiX library
find_library(PASTIX_LIBRARY pastix
  HINTS ${PASTIX_DIR} $ENV{PASTIX_DIR} ${PASTIX_DIR}/lib $ENV{PASTIX_DIR}/lib
  PATH_SUFFIXES install
  DOC "The PaStiX library"
  )

# Check for rt library
find_library(RT_LIBRARY rt
  DOC "The RT library"
  )

# Check for hwloc header
find_library(RT_LIBRARY rt
  DOC "The RT library"
  )

# Check for hwloc header
find_path(HWLOC_INCLUDE_DIRS pastix.h
  DOC "Directory where the hwloc header is located"
 )

# Check for hwloc library
find_library(HWLOC_LIBRARY hwloc
  DOC "The hwloc library"
  )

# Add BLAS libs if BLAS has been found
set(CMAKE_LIBRARY_PATH ${BLAS_DIR}/lib $ENV{BLAS_DIR}/lib ${CMAKE_LIBRARY_PATH})
find_package(BLAS)

# Collect libraries
set(PASTIX_LIBRARIES ${PASTIX_LIBRARY} ${RT_LIBRARY} ${HWLOC_LIBRARY} ${BLAS_LIBRARIES})

find_program(GFORTRAN_EXECUTABLE gfortran)
if (GFORTRAN_EXECUTABLE)
  execute_process(COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.so
    OUTPUT_VARIABLE GFORTRAN_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (EXISTS "${GFORTRAN_LIBRARY}")
    set(PASTIX_LIBRARIES ${PASTIX_LIBRARIES} ${GFORTRAN_LIBRARY})
  endif()
endif()


mark_as_advanced(
  PASTIX_INCLUDE_DIRS
  PASTIX_LIBRARY
  PASTIX_LIBRARIES
  )

# Try to run a test program that uses PaStiX
if (PASTIX_INCLUDE_DIRS AND PASTIX_LIBRARIES)

  set(CMAKE_REQUIRED_INCLUDES  ${PASTIX_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${PASTIX_LIBRARIES})

  # Add MPI variables if MPI has been found
  if (MPI_FOUND)
    set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS     "${CMAKE_REQUIRED_FLAGS} ${MPI_COMPILE_FLAGS}")
  endif()

  # Add SCOTCH variables if SCOTCH has been found
  if (SCOTCH_FOUND)
    set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${SCOTCH_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${SCOTCH_LIBRARIES})
  endif()

  # Build and run test program
  include(CheckCSourceRuns)
  check_c_source_runs("
/* Test program pastix */

#define MPICH_IGNORE_CXX_SEEK 1
#include <stdint.h>
#include <mpi.h>
#include <pastix.h>

int main()
{
  pastix_int_t iparm[IPARM_SIZE];
  double       dparm[DPARM_SIZE];
  int i = 0;
  for (i = 0; i < IPARM_SIZE; ++i)
    iparm[i] = 0;
  for (i = 0; i < DPARM_SIZE; ++i)
    dparm[i] = 0.0;

  // Set default parameters
  pastix_initParam(iparm, dparm);

  return 0;
}
" PASTIX_TEST_RUNS)

endif()
# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PASTIX
  "PaStiX could not be found. Be sure to set PASTIX_DIR."
  PASTIX_LIBRARIES PASTIX_INCLUDE_DIRS PASTIX_TEST_RUNS)
