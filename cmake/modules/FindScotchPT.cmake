# - Try to find SCOTCH
# Once done this will define
#
#  SCOTCH_FOUND        - system has found SCOTCH
#  SCOTCH_INCLUDE_DIRS - include directories for SCOTCH
#  SCOTCH_LIBARIES     - libraries for SCOTCH

set(ScotchPT_FOUND 0)

message(STATUS "Checking for package 'SCOTCH-PT'")

# Check for header file
find_path(SCOTCH_INCLUDE_DIRS ptscotch.h
  HINTS ${SCOTCH_DIR}/include $ENV{SCOTCH_DIR}/include
  PATH_SUFFIXES scotch
  DOC "Directory where the SCOTCH-PT header is located"
  )

# Check for library
#find_library(SCOTCH_LIBRARY
#  NAMES scotch
#  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
#  DOC "The SCOTCH library"
#  )

#find_library(SCOTCHERR_LIBRARY
#  NAMES scotcherr
#  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
#  DOC "The SCOTCH-ERROR library"
#  )

# Check for ptscotch
find_library(PTSCOTCH_LIBRARY
  NAMES ptscotch
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
  DOC "The PTSCOTCH library"
  )

# Check for ptscotcherr
find_library(PTSCOTCHERR_LIBRARY
  NAMES ptscotcherr
  HINTS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
  DOC "The PTSCOTCH-ERROR library"
  )

set(SCOTCH_LIBRARIES ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY})

# Try compiling and running test program
if (SCOTCH_INCLUDE_DIRS AND SCOTCH_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${SCOTCH_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${SCOTCH_LIBRARIES})

  # Add MPI variables if MPI has been found
  if (MPI_FOUND)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${MPI_COMPILE_FLAGS}")
  endif()
  message(STATUS "CMAKE_REQUIRED_INCLUDES = ${CMAKE_REQUIRED_INCLUDES}")

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
#include <stdio.h>
#include <scotch.h>
#include <iostream>
int main() {
  SCOTCH_Dgraph dgrafdat;

  if (SCOTCH_dgraphInit(&dgrafdat, MPI_COMM_WORLD) != 0) {
    std::cout << \"failure\";
  } else {
    std::cout << \"success\";
  }
  return 0;
}
" SCOTCH_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCOTCH
  "SCOTCH could not be found. Be sure to set SCOTCH_DIR."
  SCOTCH_LIBRARIES SCOTCH_INCLUDE_DIRS SCOTCH_TEST_RUNS)
