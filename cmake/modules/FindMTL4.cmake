# - Try to find MTL4
# Once done this will define
#
#  MTL4_FOUND       - system has MTL4
#  MTL4_INCLUDE_DIRS - include directories for MTL4

message(STATUS "Checking for package 'MTL4'")

# Check for header file
find_path(MTL4_INCLUDE_DIRS boost/numeric/mtl/mtl.hpp
  PATHS ${MTL4_DIR} $ENV{MTL4_DIR}
  DOC "Directory where the MTL4 header is located"
  )

# Try compiling and running test program
if (MTL4_INCLUDE_DIRS)

  # Find Boost, needed by MTL4
  set(BOOST_ROOT $ENV{BOOST_DIR})
  set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0)
  find_package(Boost REQUIRED)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES "${MTL4_INCLUDE_DIRS};${Boost_INCLUDE_DIR}")

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>
int main()
{
  mtl::dense_vector<double> x(10);
  int size = mtl::num_rows(x);
  return 0;
}
" MTL4_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MTL4
  "MTL4 could not be found. Be sure to set MTL4_DIR"
  MTL4_INCLUDE_DIRS MTL4_TEST_RUNS)
