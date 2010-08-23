# - Try to find CGAL
# Once done this will define
#
#  CGAL_FOUND        - system has CGAL
#  CGAL_INCLUDE_DIRS - include directories for CGAL
#  CGAL_LIBRARIES    - libraries for CGAL
#  CGAL_DEFINITIONS  - compiler flags for CGAL

message(STATUS "Checking for package 'CGAL'")

# Call CGAL supplied CMake script
find_package(CGAL PATHS $ENV{CGAL_DIR}/lib)

# Set variables
set(CGAL_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS})
set(CGAL_LIBRARIES    "${CGAL_LIBRARY};${CGAL_3RD_PARTY_LIBRARIES}")

# Check if we need to add special compiler option for GNU compilers
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CGAL_DEFINITIONS "-frounding-math")
endif()

# Try compiling and running test program
if (CGAL_INCLUDE_DIRS AND CGAL_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${CGAL_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${CGAL_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${CGAL_DEFINITIONS})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
// Example adapted from CGAL documentation

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double> K;

typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;

typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

int main()
{
  Point a(1.0, 0.0, 0.0);
  Point b(0.0, 1.0, 0.0);
  Point c(0.0, 0.0, 1.0);
  Point d(0.0, 0.0, 0.0);

  std::vector<Triangle> triangles;
  triangles.push_back(Triangle(a,b,c));
  triangles.push_back(Triangle(a,b,d));
  triangles.push_back(Triangle(a,d,c));

  Tree tree(triangles.begin(),triangles.end());
  Point point_query(2.0, 2.0, 2.0);
  Point closest_point = tree.closest_point(point_query);

  return 0;
}

" CGAL_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGAL.dolfin
  "CGAL could not be found. Be sure to set CGAL_DIR"
  CGAL_INCLUDE_DIRS CGAL_LIBRARIES CGAL_TEST_RUNS)
