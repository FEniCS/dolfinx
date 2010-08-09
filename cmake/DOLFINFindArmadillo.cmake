if(NOT DOLFIN_ARMADILLO_FOUND)
  message(STATUS "checking for package 'Armadillo'")

  find_path(DOLFIN_ARMADILLO_INCLUDE_DIR armadillo
    /usr/include
    /usr/local/include
    DOC "Directory where the Armadillo header file is located"
    )
  mark_as_advanced(DOLFIN_ARMADILLO_INCLUDE_DIR)
  
  find_library(DOLFIN_ARMADILLO_LIBRARY armadillo
    DOC "The Armadillo library"
    )
  mark_as_advanced(DOLFIN_ARMADILLO_LIBRARY)

  if(DOLFIN_ARMADILLO_INCLUDE_DIR AND DOLFIN_ARMADILLO_LIBRARY)
    include(CheckCXXSourceRuns)
  
    set(CMAKE_REQUIRED_INCLUDES ${DOLFIN_ARMADILLO_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${DOLFIN_ARMADILLO_LIBRARY})
  
    try_run(
      ARMADILLO_CONFIG_TEST_VERSION_EXITCODE
      ARMADILLO_CONFIG_TEST_VERSION_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}
      ${DOLFIN_CMAKE_DIR}/config_tests/armadillo_config_test_version.cpp
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
  
    set(DOLFIN_ARMADILLO_FOUND 1 CACHE TYPE BOOL)
  endif(DOLFIN_ARMADILLO_INCLUDE_DIR AND DOLFIN_ARMADILLO_LIBRARY)
  
  if(DOLFIN_ARMADILLO_FOUND)
    message(STATUS "  found Armadillo, version ${ARMADILLO_VERSION}")
  else(DOLFIN_ARMADILLO_FOUND)
    message(STATUS "  package 'Armadillo' not found")
  endif(DOLFIN_ARMADILLO_FOUND)
endif(NOT DOLFIN_ARMADILLO_FOUND)
