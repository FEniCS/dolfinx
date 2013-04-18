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
// Last changed: 2013-04-18

#ifndef __INTERSECTION_H
#define __INTERSECTION_H

#include <boost/shared_ptr.hpp>
#include "MeshPointIntersection.h"

namespace dolfin
{

  class Mesh;
  class Point;

  /// Compute and return intersection between _Mesh_ and _Point_.
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh to be intersected.
  ///     point (_Point_)
  ///         The point to be intersected.
  ///
  /// *Returns*
  ///     _MeshPointIntersection_
  ///         The intersection data.
  boost::shared_ptr<const MeshPointIntersection>
  intersect(const Mesh& mesh, const Point& point)
  {
    return boost::shared_ptr<const MeshPointIntersection>
      (new MeshPointIntersection(mesh, point));
  }

}

#endif
