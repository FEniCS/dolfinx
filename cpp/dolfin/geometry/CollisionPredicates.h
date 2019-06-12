// Copyright (C) 2014-2016 Anders Logg, August Johansson and Benjamin Kehlet
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace dolfin
{
namespace mesh
{
class MeshEntity;
}

namespace geometry
{

/// This class implements algorithms for detecting pairwise
/// collisions between mesh entities of varying dimensions.

class CollisionPredicates
{
public:
  //--- High-level collision detection predicates ---

  /// Check whether entity collides with point.
  ///
  /// @param     entity (_MeshEntity_)
  ///         The entity.
  /// @param    point (_Point_)
  ///         The point.
  ///
  /// @returns    bool
  ///         True iff entity collides with cell.
  static bool collides(const mesh::MeshEntity& entity,
                       const Eigen::Vector3d& point);

  /// Check whether two entities collide.
  ///
  /// @param     entity_0 (_MeshEntity_)
  ///         The first entity.
  /// @param    entity_1 (_MeshEntity_)
  ///         The second entity.
  ///
  /// @returns    bool
  ///         True iff entity collides with cell.
  static bool collides(const mesh::MeshEntity& entity_0,
                       const mesh::MeshEntity& entity_1);

  //--- Low-level collision detection predicates ---

  /// Check whether segment p0-p1 collides with point
  static bool collides_segment_point(const Eigen::Vector3d& p0,
                                     const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& point,
                                     std::size_t gdim);

  /// Check whether segment p0-p1 collides with point (1D version)
  static bool collides_segment_point_1d(double p0, double p1, double point);

  /// Check whether segment p0-p1 collides with point (2D version)
  static bool collides_segment_point_2d(const Eigen::Vector3d& p0,
                                        const Eigen::Vector3d& p1,
                                        const Eigen::Vector3d& point);

  /// Check whether segment p0-p1 collides with point (3D version)
  static bool collides_segment_point_3d(const Eigen::Vector3d& p0,
                                        const Eigen::Vector3d& p1,
                                        const Eigen::Vector3d& point);

  /// Check whether segment p0-p1 collides with segment q0-q1
  static bool collides_segment_segment(const Eigen::Vector3d& p0,
                                       const Eigen::Vector3d& p1,
                                       const Eigen::Vector3d& q0,
                                       const Eigen::Vector3d& q1,
                                       std::size_t gdim);

  /// Check whether segment p0-p1 collides with segment q0-q1 (1D version)
  static bool collides_segment_segment_1d(double p0, double p1, double q0,
                                          double q1);

  /// Check whether segment p0-p1 collides with segment q0-q1 (2D version)
  static bool collides_segment_segment_2d(const Eigen::Vector3d& p0,
                                          const Eigen::Vector3d& p1,
                                          const Eigen::Vector3d& q0,
                                          const Eigen::Vector3d& q1);

  /// Check whether segment p0-p1 collides with segment q0-q1 (3D version)
  static bool collides_segment_segment_3d(const Eigen::Vector3d& p0,
                                          const Eigen::Vector3d& p1,
                                          const Eigen::Vector3d& q0,
                                          const Eigen::Vector3d& q1);

  /// Check whether triangle p0-p1-p2 collides with point
  static bool collides_triangle_point(const Eigen::Vector3d& p0,
                                      const Eigen::Vector3d& p1,
                                      const Eigen::Vector3d& p2,
                                      const Eigen::Vector3d& point,
                                      std::size_t gdim);

  /// Check whether triangle p0-p1-p2 collides with point (2D version)
  static bool collides_triangle_point_2d(const Eigen::Vector3d& p0,
                                         const Eigen::Vector3d& p1,
                                         const Eigen::Vector3d& p2,
                                         const Eigen::Vector3d& point);

  /// Check whether triangle p0-p1-p2 collides with point (3D version)
  static bool collides_triangle_point_3d(const Eigen::Vector3d& p0,
                                         const Eigen::Vector3d& p1,
                                         const Eigen::Vector3d& p2,
                                         const Eigen::Vector3d& point);

  /// Check whether triangle p0-p1-p2 collides with segment q0-q1
  static bool collides_triangle_segment(const Eigen::Vector3d& p0,
                                        const Eigen::Vector3d& p1,
                                        const Eigen::Vector3d& p2,
                                        const Eigen::Vector3d& q0,
                                        const Eigen::Vector3d& q1,
                                        std::size_t gdim);

  /// Check whether triangle p0-p1-p2 collides with segment q0-q1 (2D version)
  static bool collides_triangle_segment_2d(const Eigen::Vector3d& p0,
                                           const Eigen::Vector3d& p1,
                                           const Eigen::Vector3d& p2,
                                           const Eigen::Vector3d& q0,
                                           const Eigen::Vector3d& q1);

  /// Check whether triangle p0-p1-p2 collides with segment q0-q1 (3D version)
  static bool collides_triangle_segment_3d(const Eigen::Vector3d& p0,
                                           const Eigen::Vector3d& p1,
                                           const Eigen::Vector3d& p2,
                                           const Eigen::Vector3d& q0,
                                           const Eigen::Vector3d& q1);

  /// Check whether triangle p0-p1-p2 collides with triangle q0-q1-q2
  static bool collides_triangle_triangle(
      const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
      const Eigen::Vector3d& p2, const Eigen::Vector3d& q0,
      const Eigen::Vector3d& q1, const Eigen::Vector3d& q2, std::size_t gdim);

  /// Check whether triangle p0-p1-p2 collides with triangle q0-q1-q2 (2D
  /// version)
  static bool collides_triangle_triangle_2d(const Eigen::Vector3d& p0,
                                            const Eigen::Vector3d& p1,
                                            const Eigen::Vector3d& p2,
                                            const Eigen::Vector3d& q0,
                                            const Eigen::Vector3d& q1,
                                            const Eigen::Vector3d& q2);

  /// Check whether triangle p0-p1-p2 collides with triangle q0-q1-q2 (3D
  /// version)
  static bool collides_triangle_triangle_3d(const Eigen::Vector3d& p0,
                                            const Eigen::Vector3d& p1,
                                            const Eigen::Vector3d& p2,
                                            const Eigen::Vector3d& q0,
                                            const Eigen::Vector3d& q1,
                                            const Eigen::Vector3d& q2);

  /// Check whether tetrahedron p0-p1-p2-p3 collides with point
  static bool collides_tetrahedron_point_3d(const Eigen::Vector3d& p0,
                                            const Eigen::Vector3d& p1,
                                            const Eigen::Vector3d& p2,
                                            const Eigen::Vector3d& p3,
                                            const Eigen::Vector3d& point);

  /// Check whether tetrahedron p0-p1-p2-p3 collides with segment q0-q1
  static bool collides_tetrahedron_segment_3d(const Eigen::Vector3d& p0,
                                              const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& p2,
                                              const Eigen::Vector3d& p3,
                                              const Eigen::Vector3d& q0,
                                              const Eigen::Vector3d& q1);

  /// Check whether tetrahedron p0-p1-p2-p3 collides with triangle q0-q1-q2
  static bool collides_tetrahedron_triangle_3d(const Eigen::Vector3d& p0,
                                               const Eigen::Vector3d& p1,
                                               const Eigen::Vector3d& p2,
                                               const Eigen::Vector3d& p3,
                                               const Eigen::Vector3d& q0,
                                               const Eigen::Vector3d& q1,
                                               const Eigen::Vector3d& q2);

  /// Check whether tetrahedron p0-p1-p2-p3 collides with tetrahedron q0-q1-q2
  static bool collides_tetrahedron_tetrahedron_3d(
      const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
      const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
      const Eigen::Vector3d& q0, const Eigen::Vector3d& q1,
      const Eigen::Vector3d& q2, const Eigen::Vector3d& q3);
};
} // namespace geometry
} // namespace dolfin
