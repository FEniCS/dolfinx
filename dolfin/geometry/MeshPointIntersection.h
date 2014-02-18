// Copyright (C) 2013 Anders Logg
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
// First added:  2013-04-18
// Last changed: 2013-05-30

#ifndef __MESH_POINT_INTERSECTION_H
#define __MESH_POINT_INTERSECTION_H

#include <vector>
#include <memory>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class Point;

  /// This class represents an intersection between a _Mesh_ and a
  /// _Point_. The resulting intersection is stored as a list of zero
  /// or more cells.

  class MeshPointIntersection
  {
  public:

    /// Compute intersection between mesh and point
    MeshPointIntersection(const Mesh& mesh,
                          const Point& point);

    /// Destructor
    ~MeshPointIntersection();

    /// Return the list of (local) indices for intersected cells
    const std::vector<unsigned int>& intersected_cells() const
    { return _intersected_cells; }

  private:

    // The list of (local) indices for intersected cells
    std::vector<unsigned int> _intersected_cells;

  };

}

#endif
