set(MTL4_FOUND 0)

message(STATUS "checking for package 'MTL4'")

find_path(MTL4_INCLUDE_DIR boost/numeric/mtl/mtl.hpp
  $ENV{MTL4_DIR}
  /usr/local/include
  /usr/include
  DOC "Directory where the MTL4 header directory is located"
  )

if(MTL4_INCLUDE_DIR)
  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_INCLUDES ${MTL4_INCLUDE_DIR})
  check_cxx_source_runs("
#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>
int main()
{
  mtl::dense_vector<double> x(10);
  int size = mtl::num_rows(x);
  return 0;
}
"
  MTL4_TEST_RUNS)

  if(NOT MTL4_TEST_RUNS)
    message("MTL4 was found but a test program could not be run.")
  endif(NOT MTL4_TEST_RUNS)

endif(MTL4_INCLUDE_DIR)

if(MTL4_TEST_RUNS)
  message(STATUS "  found package MTL4")
  set(MTL4_FOUND 1)
  include_directories(${MTL4_INCLUDE_DIR})
  add_definitions(-DHAS_MTL4)
else(MTL4_TEST_RUNS)
  message(STATUS "  package 'MTL4' could not be configured.")
endif(MTL4_TEST_RUNS)

