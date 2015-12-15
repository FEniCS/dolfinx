// Copyright (C) 2012-2013 Anders Logg
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
// Last changed: 2013-06-27

#ifndef __MESH_TRANSFORMATION_H
#define __MESH_TRANSFORMATION_H

namespace dolfin
{

  class Mesh;
  class Point;

  /// This class implements various transformations of the coordinates
  /// of a mesh.

class MeshTransformation
{
public:

  /// Translate mesh according to a given vector.
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh
  ///     point (Point)
  ///         The vector defining the translation.
  static void translate(Mesh& mesh, const Point& point);

  /// Rescale mesh by a given scaling factor with respect to a center
  /// point.
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh
  ///      scale (double)
  ///         The scaling factor.
  ///      center (Point)
  ///         The center of the scaling.
  static void rescale(Mesh& mesh, const double scale, const Point& center);

  /// Rotate mesh around a coordinate axis through center of mass
  /// of all mesh vertices
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh.
  ///     angle (double)
  ///         The number of degrees (0-360) of rotation.
  ///     axis (std::size_t)
  ///         The coordinate axis around which to rotate the mesh.
  static void rotate(Mesh& mesh, double angle, std::size_t axis);

  /// Rotate mesh around a coordinate axis through a given point
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh.
  ///     angle (double)
  ///         The number of degrees (0-360) of rotation.
  ///     axis (std::size_t)
  ///         The coordinate axis around which to rotate the mesh.
  ///     point (_Point_)
  ///         The point around which to rotate the mesh.
  static void rotate(Mesh& mesh, double angle, std::size_t axis,
                     const Point& p);

};

}

#endif
