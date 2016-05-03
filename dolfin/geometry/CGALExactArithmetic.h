// Copyright (C) 2016 Benjamin Kehlet, August Johansson, and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2016-05-03
// Last changed: 2016-05-03
//
// Developer note:
//
// This file contains reference implementations of collision detection
// algorithms using exact arithmetic with CGAL. It is not included in
// a normal build but is used as a reference for verification and
// debugging of the inexact DOLFIN collision detection algorithms.
// Enable by setting the flag DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC.

namespace dolfin
{

#ifndef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC

  // Macro that just returns the result when CGAL is not enabled
#define CHECK_CGAL(RESULT, ...) RESULT

#else

  // FIXME: Remove this
#include <dolfin/geometry/dolfin_cgal_tools.h>

  // Compare results and report any errors
  bool check_cgal(bool result, bool result_cgal)
  {
    if (results != result_cgal)
      error("DOLFIN test does not match CGAL reference test.");
    return result;
  }

  // Macro that calls the corresponding CGAL version of the function
  // and compares RESULT with the result from the CGAL version.
#define CHECK_CGAL(RESULT, ...) check_cgal(result, __FUNCTION__(__VA_ARGS__))

  //---------------------------------------------------------------------------
  // Reference implementations using CGAL exact arithmetic
  //---------------------------------------------------------------------------
  void cgal_collides_interval_interval(const MeshEntity& interval_0,
                                       const MeshEntity& interval_1)
  {
    const MeshGeometry& geometry_0 = interval_0.mesh().geometry();
    const MeshGeometry& geometry_1 = interval_1.mesh().geometry();
    const unsigned int* vertices_0 = interval_0.entities(0);
    const unsigned int* vertices_1 = interval_1.entities(0);

    const Point a(geometry_0.point(vertices_0[0])[0],
                  geometry_0.point(vertices_0[0])[1]);
    const Point b(geometry_0.point(vertices_0[1])[0],
                  geometry_0.point(vertices_0[1])[1]);
    const Point c(geometry_1.point(vertices_1[0])[0],
                  geometry_1.point(vertices_1[0])[1]);
    const Point d(geometry_1.point(vertices_1[1])[0],
                  geometry_1.point(vertices_1[1])[1]);

    return CGAL::do_intersect(cgaltools::convert(a, b),
                              cgaltools::convert(c, d));
  }
  //---------------------------------------------------------------------------

#endif

}
