# - Try to find CGAL
# Once done this will define
#
#  CGAL_FOUND        - system has CGAL
#  CGAL_INCLUDE_DIRS - include directories for CGAL
#  CGAL_LIBRARIES    - libraries for CGAL
#  CGAL_DEFINITIONS  - compiler flags for CGAL

#=============================================================================
# Copyright (C) 2010-2011 Anders Logg, Johannes Ring and Garth N. Wells
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

message(STATUS "Checking for package 'CGAL'")

# Blank out CGAL_FIND_VERSION temporarily or else find_package(CGAL ...)
# (below) will fail.
set(CGAL_FIND_VERSION_TMP ${CGAL_FIND_VERSION})
set(CGAL_FIND_VERSION "")

# Call CGAL supplied CMake script
find_package(CGAL
  HINTS
  ${CGAL_DIR}
  $ENV{CGAL_DIR}
  PATH_SUFFIXES lib cmake/modules lib/cmake lib/CGAL)

# Restore CGAL_FIND_VERSION
set(CGAL_FIND_VERSION ${CGAL_FIND_VERSION_TMP})

if (CGAL_FIND_VERSION)
  # Check if version found is >= required version
  if (NOT "${CGAL_VERSION}" VERSION_LESS "${CGAL_FIND_VERSION}")
    set(CGAL_VERSION_OK TRUE)
  endif()
else()
  # No specific version of CGAL is requested
  set(CGAL_VERSION_OK TRUE)
endif()

# Set variables
set(CGAL_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS} ${CGAL_3RD_PARTY_INCLUDE_DIRS})
set(CGAL_LIBRARIES ${CGAL_LIBRARY} ${CGAL_3RD_PARTY_LIBRARIES})

# Try compiling and running test program
if (CGAL_INCLUDE_DIRS AND CGAL_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${CGAL_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${CGAL_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${CGAL_CXX_FLAGS_INIT})

  # Add all previusly found Boost libraries - CGAL doesn't appear to supply
  # all necessary Boost libs (test with Boost 1.50 + CGAL 4.0.2)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${Boost_LIBRARIES})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
// CGAL test program from Andre Massing

#include <CGAL/AABB_tree.h> // *Must* be inserted before kernel!
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <CGAL/Simple_cartesian.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Bbox_3.h>
#include <CGAL/Point_3.h>

#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>

typedef CGAL::Simple_cartesian<double> SCK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron_3;

typedef SCK::FT FT;
typedef SCK::Ray_3 Ray;
typedef SCK::Line_3 Line;
typedef SCK::Point_3 Point;
typedef SCK::Triangle_3 Triangle;

typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<SCK,Iterator> Primitive;
typedef CGAL::AABB_traits<SCK, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

typedef Nef_polyhedron_3::Aff_transformation_3 Aff_transformation_3;
typedef Nef_polyhedron_3::Plane_3 Plane_3;
typedef Nef_polyhedron_3::Vector_3 Vector_3;
typedef Nef_polyhedron_3::Point_3 Point_3;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron_3;

int main()
{
  //CGAL exact points
  Point_3 p1(0,0,0);
  Point_3 p2(1,0,0);
  Point_3 p3(0,1,0);
  Point_3 p4(0,0,1);

  Polyhedron_3 P;
  P.make_tetrahedron(p1,p2,p3,p4);
  Nef_polyhedron_3 NP(P);
  NP.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, 1, 1)));

  //Inexact points
  Point a(1.0, 0.0, 0.0);
  Point b(0.0, 1.0, 0.0);
  Point c(0.0, 0.0, 1.0);
  Point d(0.0, 0.0, 0.0);

  std::list<Triangle> triangles;
  triangles.push_back(Triangle(a,b,c));
  triangles.push_back(Triangle(a,b,d));
  triangles.push_back(Triangle(a,d,c));

  // constructs AABB tree
  Tree tree(triangles.begin(),triangles.end());

  // counts #intersections
  Ray ray_query(a,b);
  std::cout << tree.number_of_intersected_primitives(ray_query)
      << \" intersections(s) with ray query\" << std::endl;

  // compute closest point and squared distance
  Point point_query(2.0, 2.0, 2.0);
  Point closest_point = tree.closest_point(point_query);

  return 0;
}
" CGAL_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGAL
  "CGAL could not be found. Be sure to set CGAL_DIR"
  CGAL_LIBRARIES CGAL_INCLUDE_DIRS CGAL_TEST_RUNS CGAL_VERSION_OK)
