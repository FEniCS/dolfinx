// Copyright (C) 2016 Anders Logg and August Johansson, Benjamin Kehlet
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
// First added:  2016-06-01
// Last changed: 2016-06-06

#ifndef __CONVEX_TRIANGULATION
#define __CONVEX_TRIANGULATION

#include "Point.h"
#include <vector>

namespace dolfin
{

class ConvexTriangulation
{
 public:

  // Tdim independent wrapper
  static std::vector<std::vector<Point>>
    triangulate(std::vector<Point> p,
                std::size_t gdim);


  static std::vector<std::vector<Point>>
  triangulate_graham_scan(std::vector<Point> p,
                          std::size_t gdim);

  // TODO: Fix this.
  static std::vector<std::vector<Point>>
  triangulate_3d(std::vector<Point> p);

};


} // end namespace dolfin
#endif
