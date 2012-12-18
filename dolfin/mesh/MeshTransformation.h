// Copyright (C) 2012 Anders Logg
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
// First added:  2012-01-16
// Last changed: 2012-01-16

#ifndef __MESH_TRANSFORMATION_H
#define __MESH_TRANSFORMATION_H

namespace dolfin
{

   class Mesh;

  /// This class implements various transformations of the coordinates
  /// of a mesh.

  // FIXME: Consider adding other transformations, such as for example
  // translation and stretching.

  class MeshTransformation
  {
  public:

    /// Rotate mesh around a coordinate axis through center of mass
    /// of all mesh vertices
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    ///     angle (double)
    ///         The number of degrees (0-360) of rotation
    ///     axis (std::size_t)
    ///         The coordinate axis around which to rotate the mesh
    static void rotate(Mesh& mesh, double angle, std::size_t axis);

    /// Rotate mesh around a coordinate axis through a given point
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    ///     angle (double)
    ///         The number of degrees (0-360) of rotation
    ///     axis (std::size_t)
    ///         The coordinate axis around which to rotate the mesh
    ///     point (_Point_)
    ///         The point around which to rotate the mesh
    static void rotate(Mesh& mesh, double angle, std::size_t axis, const Point& p);

  };

}

#endif
