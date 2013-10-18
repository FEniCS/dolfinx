// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-10-18
// Last changed:

#ifndef __ELLIPSOID_MESH_H
#define __ELLIPSOID_MESH_H

#include <vector>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  class Point;

  /// Tetrahedral mesh an ellipsoid. CGAL is used to generate the
  /// mesh.



  class EllipsoidMesh : public Mesh
  {
  public:

    /// Create an unstructured _Mesh_ of an ellipsoid
    ///
    /// *Arguments*
    ///     center (_Point))
    ///         Center of the ellipsoid
    ///     dims (std::vector<double>)
    ///         Axes lengths
    ///     cell_size (double)
    ///         Cell size measure
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         // Create elliposid mesh
    ///         std::vector<double> dims(3);
    //          dims[0] = 1.0; dims[1] = 2.0; dims[2] = 1.5;
    ///         EllipsoidMesh mesh(Point(1.0, 2.0, -1.0), dims, 0.2);
    ///
    EllipsoidMesh(Point p, std::vector<double> ellipsoid_dims,
                  double cell_size);

  };

  class SphereMesh : public EllipsoidMesh
  {
  public:

    /// Create an unstructured _Mesh_ of sphere
    ///
    /// *Arguments*
    ///     center (_Point_)
    ///         Center of the ellipsoid
    ///     radius (double)
    ///         Axes lengths
    ///     cell_size (double)
    ///         Cell size measure
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         // Create sphere with center (1.0, 2.0, -1.0) and
    ///         // radius 3.0. Cell size is 0.2.
    ///         SphereMesh mesh(Point(1.0, 2.0, -1.0), 3.0, 0.2);
    ///
    SphereMesh(Point p, double radius, double cell_size)
      : EllipsoidMesh(p, std::vector<double>(3, radius), cell_size) {}

  };

}

#endif
