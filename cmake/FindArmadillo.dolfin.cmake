# Try to find Armadillo - the streamlined C++ linear algebra library
#
# This module defines
# ARMADILLO_FOUND - system has Armadillo
# ARMADILLO_INCLUDE_DIR - the Armadillo include directory
# ARMADILLO_LIBRARY - the library needed to use Armadillo
# ARMADILLO_VERSION - the Armadillo version string (MAJOR.MINOR.PATCH)
#
# Setting these changes the behavior of the search
# ARMADILLO_DIR - directory in which Armadillo resides

if(NOT ARMADILLO_FOUND)
  message(STATUS "Checking for package 'Armadillo'")

  find_path(ARMADILLO_INCLUDE_DIR
    NAMES armadillo
    HINTS $ENV{ARMADILLO_DIR}
    PATHS /usr/local /opt/local /sw
    PATH_SUFFIXES include
    DOC "Directory where the Armadillo header file is located"
    )
  mark_as_advanced(ARMADILLO_INCLUDE_DIR)

  find_library(ARMADILLO_LIBRARY
    NAMES armadillo
    HINTS $ENV{ARMADILLO_DIR}
    PATHS /usr/local /opt/local /sw
    PATH_SUFFIXES lib lib64
    DOC "The Armadillo library"
    )
  mark_as_advanced(ARMADILLO_LIBRARY)

  if(ARMADILLO_INCLUDE_DIR AND ARMADILLO_LIBRARY)
    include(CheckCXXSourceRuns)

    # These are needed for the try_run and check_cxx_source_runs commands below
    set(CMAKE_REQUIRED_INCLUDES ${ARMADILLO_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${ARMADILLO_LIBRARY})

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
"
      )

    try_run(
      ARMADILLO_CONFIG_TEST_VERSION_EXITCODE
      ARMADILLO_CONFIG_TEST_VERSION_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${ARMADILLO_CONFIG_TEST_VERSION_CPP}
      RUN_OUTPUT_VARIABLE OUTPUT
      )

    if(ARMADILLO_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)
      set(ARMADILLO_VERSION ${OUTPUT} CACHE TYPE STRING)
    endif(ARMADILLO_CONFIG_TEST_VERSION_EXITCODE EQUAL 0)

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
      ARMADILLO_TEST_RUNS)

    if(NOT ARMADILLO_TEST_RUNS)
      message(FATAL_ERROR "Unable to compile and run Armadillo test program.")
    endif(NOT ARMADILLO_TEST_RUNS)

    set(ARMADILLO_FOUND 1 CACHE TYPE BOOL)
  endif(ARMADILLO_INCLUDE_DIR AND ARMADILLO_LIBRARY)

  if(ARMADILLO_FOUND)
    message(STATUS "   Found package Armadillo, version ${ARMADILLO_VERSION}")
  else(ARMADILLO_FOUND)
    message("   Unable to configure package 'Armadillo'")
  endif(ARMADILLO_FOUND)
endif(NOT ARMADILLO_FOUND)
