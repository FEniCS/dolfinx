// Copyright (C) 2014 Anders Logg
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
// First added:  2014-02-03
// Last changed: 2014-02-03

#include <vector>
#include <dolfin/log/log.h>

#ifndef __COLLISION_DETECTION_H
#define __COLLISION_DETECTION_H

namespace dolfin
{

  // Forward declarations
  class MeshEntity;

  /// This class implements algorithms for detecting pairwise
  /// collisions between mesh entities of varying dimensions.

  class CollisionDetection
  {
  public:

    /// Check whether tetrahedron collides with point.
    ///
    /// *Arguments*
    ///     tetrahedron (_MeshEntity_)
    ///         The tetrahedron.
    ///     point (_Point_)
    ///         The point.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_tetrahedron_point(const MeshEntity& tetrahedron,
                                           const Point& point);

    /// Check whether tetrahedron collides with triangle.
    ///
    /// *Arguments*
    ///     tetrahedron (_MeshEntity_)
    ///         The tetrahedron.
    ///     triangle (_MeshEntity_)
    ///         The triangle.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_tetrahedron_triangle(const MeshEntity& tetrahedron,
                                              const MeshEntity& triangle);

    /// Check whether tetrahedron collides with tetrahedron.
    ///
    /// *Arguments*
    ///     tetrahedron_0 (_MeshEntity_)
    ///         The first tetrahedron.
    ///     tetrahedron_1 (_MeshEntity_)
    ///         The second tetrahedron.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_tetrahedron_tetrahedron(const MeshEntity& tetrahedron_0,
                                                 const MeshEntity& tetrahedron_1);

  private:

    // Put private helper functions here and remove this comment.

  };

}

#endif
