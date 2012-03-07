// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-02
// Last changed:

#ifndef __DOLFIN_TRIANGULATEGENERATOR_H
#define __DOLFIN_TRIANGULATEGENERATOR_H

#ifdef HAS_CGAL

#include <vector>

namespace dolfin
{

  class Mesh;
  class Point;

  /// Create mesh from a triangulation of points

  class Triangulate
  {
  public:

    /// Create mesh from a triangulation of points
    static void triangulate(Mesh& mesh, const std::vector<Point>& vertices,
                            unsigned int gdim);

  private:

    // Create 2D mesh from a triangulation of points
    static void triangulate2D(Mesh& mesh, const std::vector<Point>& vertices);

    // Create 3D mesh from a triangulation of points
    static void triangulate3D(Mesh& mesh, const std::vector<Point>& vertices);

  };

}

#endif
#endif
