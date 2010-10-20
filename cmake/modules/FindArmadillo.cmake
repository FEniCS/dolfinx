# - Try to find PETSc
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

message(STATUS "Checking for package 'Armadillo'")

# FIXME: Look for LAPACK libraries. Required on some platforms. BLAS too?
#find_package(LAPACK REQUIRED)

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
mark_as_advanced(ARMADILLO_LIBRARIES)

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
  if (NOT Boost_INCLUDE_DIR)
    set(BOOST_ROOT $ENV{BOOST_DIR})
    set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0)
    find_package(Boost REQUIRED)
  endif()

  # These are needed for the try_run and check_cxx_source_runs commands below
  set(CMAKE_REQUIRED_INCLUDES ${ARMADILLO_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
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
    RUN_OUTPUT_VARIABLE OUTPUT
    )

  if (ARMADILLO_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
    set(ARMADILLO_VERSION ${OUTPUT} CACHE TYPE STRING)
  endif()

  # Check that C++ program runs
  check_cxx_source_compiles("
#include <armadillo>
int main()
{
 arma::mat A = arma::rand(4, 4);
 arma::vec b = arma::rand(4);
 arma::vec x = arma::solve(A, b);
 return 0;
}
"
    ARMADILLO_TEST_RUNS)

  # If program runs, trying adding LAPACK library and test again
  if(NOT ARMADILLO_TEST_RUNS)
    find_package(LAPACK)
    if (LAPACK_LIBRARIES)
      set(CMAKE_REQUIRED_LIBRARIES ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES})
      check_cxx_source_runs("
#include <armadillo>
int main()
{
 arma::mat A = arma::rand(4, 4);
 arma::vec b = arma::rand(4);
 arma::vec x = arma::solve(A, b);
 return 0;
}
"
      ARMADILLO_LAPACK_TEST_RUNS)

      # Add LAPACK libary if required
      if (ARMADILLO_LAPACK_TEST_RUNS)
        set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARIES} ${LAPACK_LIBRARIES})
      endif()

    endif()
  endif()
endif()

# Set test run to 'true' if test runs with or without LAPACK lib
if(ARMADILLO_TEST_RUNS OR ARMADILLO_LAPACK_TEST_RUNS)
  set(ARMADILLO_TEST_RUNS TRUE)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadillo
  "Armadillo could not be found. Be sure to set ARMADILLO_DIR."
  ARMADILLO_LIBRARIES ARMADILLO_INCLUDE_DIRS ARMADILLO_TEST_RUNS)
