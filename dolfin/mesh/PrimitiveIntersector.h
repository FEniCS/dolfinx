// Copyright (C) 2010 André Massing.
// Licensed under the GNU LGPL Version 2.1.
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
