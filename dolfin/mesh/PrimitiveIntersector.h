// Copyright (C) 2010 Andre Massing
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-02-09
// Last changed: 2010-03-05

#ifndef  __PRIMITIVE_INTERSECTOR_H
#define  __PRIMITIVE_INTERSECTOR_H

namespace dolfin
{
  class MeshEntity;
  class Point;

  /// This class implements an intersection detection, detecting
  /// whether two given (arbitrary) meshentities intersect.

  class PrimitiveIntersector
  {
  public:

    /// Computes whether two mesh entities intersect using an inexact
    /// geometry kernel which is fast but may suffer from floating
    /// point precision
    static bool do_intersect(const MeshEntity& entity_1,
                             const MeshEntity& entity_2);

    /// Computes whether a mesh entity and point intersect using an
    /// inexact geometry kernel which is fast but may suffer from
    /// floating point precision
    static bool do_intersect(const MeshEntity& entity_1,
                             const Point& point);

    /// Computes whether two mesh entities intersect using an exact
    /// geometry kernel which is slow but always correct
    static bool do_intersect_exact(const MeshEntity& entity_1,
                                   const MeshEntity& entity_2);

    /// Computes whether a mesh entity and point intersect using an
    /// exact geometry kernel which is slow but always correct
    static bool do_intersect_exact(const MeshEntity& entity_1,
                                   const Point& point);

  private:

    // Helper classes to deal with all combination in a N and not N*N way.
    // Just declaration, definition and instantation takes place in the corresponding cpp file, where
    // this helper function are actually needed.

    template <typename K, typename T, typename U >
    static bool do_intersect_with_kernel(const T& entity_1,
                                         const U& entity_2);

    template<typename K, typename T>
    static bool do_intersect_with_kernel(const T& entity_1,
                                         const MeshEntity& entity_2);

  };

}

#endif
