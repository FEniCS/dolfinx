// Copyright (C) 2016 Anders Logg, August Johansson and Benjamin Kehlet
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
// First added:  2016-11-21
// Last changed: 2016-11-21

#ifndef __GEOMETRYPREDICATES_H
#define __GEOMETRYPREDICATES_H

#include <vector>
#include "Point.h"
#include "CGALExactArithmetic.h"

namespace dolfin
{

  class GeometryPredicates
  {
  public:

    /// Check whether simplex is degenerate
    static bool is_degenerate(const std::vector<Point>& simplex,
			      std::size_t gdim);

    static bool is_degenerate_2d(const std::vector<Point>& simplex)
    {
      return CHECK_CGAL(_is_degenerate_2d(simplex),
			cgal_is_degenerate_2d(simplex));
    }

    static bool is_degenerate_3d(const std::vector<Point>& simplex)
    {
      return CHECK_CGAL(_is_degenerate_3d(simplex),
			cgal_is_degenerate_3d(simplex));
    }

  private:

    // Implementations of is_degenerate
    static bool _is_degenerate_2d(std::vector<Point> simplex);

    static bool _is_degenerate_3d(std::vector<Point> simplex);

  };

}

#endif
