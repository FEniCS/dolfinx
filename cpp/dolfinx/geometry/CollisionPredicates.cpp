// Copyright (C) 2014-2017 Anders Logg, August Johansson and Benjamin Kehlet
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CollisionPredicates.h"
#include "predicates.h"
#include <Eigen/Dense>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/cell_types.h>

using namespace dolfinx;
using namespace dolfinx::geometry;

/// Compute numerically stable cross product (a - c) x (b - c)
namespace
{
Eigen::Vector3d cross_product(const Eigen::Vector3d& a,
                              const Eigen::Vector3d& b,
                              const Eigen::Vector3d& c)
{
  // See Shewchuk Lecture Notes on Geometric Robustness
  double ayz[2] = {a[1], a[2]};
  double byz[2] = {b[1], b[2]};
  double cyz[2] = {c[1], c[2]};
  double azx[2] = {a[2], a[0]};
  double bzx[2] = {b[2], b[0]};
  double czx[2] = {c[2], c[0]};
  double axy[2] = {a[0], a[1]};
  double bxy[2] = {b[0], b[1]};
  double cxy[2] = {c[0], c[1]};
  Eigen::Vector3d r(_orient2d(ayz, byz, cyz), _orient2d(azx, bzx, czx),
                    _orient2d(axy, bxy, cxy));
  return r;
}
} // namespace

//-----------------------------------------------------------------------------
// High-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides(const mesh::MeshEntity& entity,
                                   const Eigen::Vector3d& point)
{
  mesh::CellType cell_type = entity.mesh().topology().cell_type();

  // Intersection is only implemented for simplex meshes
  if (!mesh::is_simplex(cell_type)
      and !(cell_type == mesh::CellType::quadrilateral))
  {
    throw std::runtime_error(
        "Cannot intersect cell and point. "
        "Intersection is only implemented for simplex meshes and quads");
  }

  // Get data
  const mesh::Geometry& g = entity.mesh().geometry();
  auto v = entity.entities(0);
  const int tdim = entity.mesh().topology().dim();
  const int gdim = entity.mesh().geometry().dim();

  // Hack to get correct indices
  auto comp_vertices = [&v](const mesh::MeshEntity& entity, int n) {
    std::array<int, 4> idx;
    for (int i = 0; i < n; ++i)
    {
      const graph::AdjacencyList<std::int32_t>& x_dofmap
          = entity.mesh().geometry().dofmap();
      const int tdim = entity.mesh().topology().dim();

      // Find attached cell
      auto e_to_c = entity.mesh().topology().connectivity(entity.dim(), tdim);
      assert(e_to_c);
      assert(e_to_c->num_links(entity.index()) > 0);
      const std::int32_t c = e_to_c->links(entity.index())[0];

      auto dofs = x_dofmap.links(c);
      auto c_to_v = entity.mesh().topology().connectivity(tdim, 0);
      assert(c_to_v);
      auto cell_vertices = c_to_v->links(c);

      const auto *it = std::find(cell_vertices.data(),
                          cell_vertices.data() + cell_vertices.rows(), v[i]);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_vertex = std::distance(cell_vertices.data(), it);
      idx[i] = dofs(local_vertex);
    }
    return idx;
  };

  // Pick correct specialized implementation
  if (cell_type == mesh::CellType::quadrilateral)
  {
    std::array<int, 4> _v = comp_vertices(entity, 4);
    return collides_quad_point_2d(g.node(_v[0]), g.node(_v[1]), g.node(_v[2]),
                                  g.node(_v[3]), point);
  }
  else if (tdim == 1 and gdim == 1)
  {
    std::array<int, 4> _v = comp_vertices(entity, 2);
    return collides_segment_point_1d(g.node(_v[0])[0], g.node(_v[1])[0],
                                     point[0]);
  }
  else if (tdim == 1 and gdim == 2)
  {
    std::array<int, 4> _v = comp_vertices(entity, 2);
    return collides_segment_point_2d(g.node(_v[0]), g.node(_v[1]), point);
  }
  else if (tdim == 1 and gdim == 3)
  {
    std::array<int, 4> _v = comp_vertices(entity, 2);
    return collides_segment_point_3d(g.node(_v[0]), g.node(_v[1]), point);
  }
  else if (tdim == 2 and gdim == 2)
  {
    std::array<int, 4> _v = comp_vertices(entity, 3);
    return collides_triangle_point_2d(g.node(_v[0]), g.node(_v[1]),
                                      g.node(_v[2]), point);
  }
  else if (tdim == 2 and gdim == 3)
  {
    std::array<int, 4> _v = comp_vertices(entity, 3);
    return collides_triangle_point_3d(g.node(_v[0]), g.node(_v[1]),
                                      g.node(_v[2]), point);
  }
  else if (tdim == 3)
  {
    std::array<int, 4> _v = comp_vertices(entity, 4);
    return collides_tetrahedron_point_3d(g.node(_v[0]), g.node(_v[1]),
                                         g.node(_v[2]), g.node(_v[3]), point);
  }
  else
  {
    throw std::runtime_error("Cannot compute entity-point collision. "
                             "Not implemented for dimensions"
                             + std::to_string(tdim) + "/"
                             + std::to_string(gdim));
    return false;
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides(const mesh::MeshEntity& entity_0,
                                   const mesh::MeshEntity& entity_1)
{
  // Intersection is only implemented for simplex meshes
  if (!mesh::is_simplex(entity_0.mesh().topology().cell_type())
      or !mesh::is_simplex(entity_1.mesh().topology().cell_type()))
  {
    throw std::runtime_error(
        "Cannot intersect cell and point. "
        "Intersection is only implemented for simplex meshes");
  }

  // Get data
  const mesh::Geometry& g0 = entity_0.mesh().geometry();
  const mesh::Geometry& g1 = entity_1.mesh().geometry();
  const std::size_t d0 = entity_0.dim();
  const std::size_t d1 = entity_1.dim();
  const int gdim = g0.dim();
  assert(gdim == g1.dim());

  // Hack to get correct indices
  auto comp_vertices = [](const mesh::MeshEntity& entity, int n) {
    std::array<int, 4> idx;
    auto v = entity.entities(0);
    for (int i = 0; i < n; ++i)
    {
      const graph::AdjacencyList<std::int32_t>& x_dofmap
          = entity.mesh().geometry().dofmap();
      const int tdim = entity.mesh().topology().dim();

      // Find attached cell
      auto e_to_c = entity.mesh().topology().connectivity(entity.dim(), tdim);
      assert(e_to_c);
      assert(e_to_c->num_links(entity.index()) > 0);
      const std::int32_t c = e_to_c->links(entity.index())[0];

      auto dofs = x_dofmap.links(c);
      auto c_to_v = entity.mesh().topology().connectivity(tdim, 0);
      assert(c_to_v);
      auto cell_vertices = c_to_v->links(c);

      const auto *it = std::find(cell_vertices.data(),
                          cell_vertices.data() + cell_vertices.rows(), v[i]);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_vertex = std::distance(cell_vertices.data(), it);
      idx[i] = dofs(local_vertex);
    }
    return idx;
  };

  // Pick correct specialized implementation
  if (d0 == 1 && d1 == 1)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 2);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 2);
    return collides_segment_segment(g0.node(_v0[0]), g0.node(_v0[1]),
                                    g1.node(_v1[0]), g1.node(_v1[1]), gdim);
  }
  else if (d0 == 1 && d1 == 2)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 2);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 3);
    return collides_triangle_segment(g1.node(_v1[0]), g1.node(_v1[1]),
                                     g1.node(_v1[2]), g0.node(_v0[0]),
                                     g0.node(_v0[1]), gdim);
  }
  else if (d0 == 2 && d1 == 1)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 3);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 2);
    return collides_triangle_segment(g0.node(_v0[0]), g0.node(_v0[1]),
                                     g0.node(_v0[2]), g1.node(_v1[0]),
                                     g1.node(_v1[1]), gdim);
  }
  else if (d0 == 2 && d1 == 2)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 3);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 3);
    return collides_triangle_triangle(g0.node(_v0[0]), g0.node(_v0[1]),
                                      g0.node(_v0[2]), g1.node(_v1[0]),
                                      g1.node(_v1[1]), g1.node(_v1[2]), gdim);
  }
  else if (d0 == 2 && d1 == 3)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 2);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 4);
    return collides_tetrahedron_triangle_3d(
        g1.node(_v1[0]), g1.node(_v1[1]), g1.node(_v1[2]), g1.node(_v1[3]),
        g0.node(_v0[0]), g0.node(_v0[1]), g0.node(_v0[2]));
  }
  else if (d0 == 3 && d1 == 2)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 4);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 3);
    return collides_tetrahedron_triangle_3d(
        g0.node(_v0[0]), g0.node(_v0[1]), g0.node(_v0[2]), g0.node(_v0[3]),
        g1.node(_v1[0]), g1.node(_v1[1]), g1.node(_v1[2]));
  }
  else if (d0 == 3 && d1 == 3)
  {
    std::array<int, 4> _v0 = comp_vertices(entity_0, 4);
    std::array<int, 4> _v1 = comp_vertices(entity_1, 4);
    return collides_tetrahedron_tetrahedron_3d(
        g0.node(_v0[0]), g0.node(_v0[1]), g0.node(_v0[2]), g0.node(_v0[3]),
        g1.node(_v1[0]), g1.node(_v1[1]), g1.node(_v1[2]), g1.node(_v1[3]));
  }
  else
  {
    throw std::runtime_error("Cannot compute entity-entity collision. "
                             "Not implemented for topological dimensions "
                             + std::to_string(d0) + " / " + std::to_string(d1)
                             + " and geometrical dimension "
                             + std::to_string(gdim));
    return false;
  }
}
//-----------------------------------------------------------------------------
// Low-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point(const Eigen::Vector3d& p0,
                                                 const Eigen::Vector3d& p1,
                                                 const Eigen::Vector3d& point,
                                                 std::size_t gdim)
{
  switch (gdim)
  {
  case 1:
    return collides_segment_point_1d(p0[0], p1[0], point[0]);
  case 2:
    return collides_segment_point_2d(p0, p1, point);
  case 3:
    return collides_segment_point_3d(p0, p1, point);
  default:
    throw std::runtime_error(
        "Cannot call collides_segment_point. Unknown "
        "dimension (only implemented for dimension 2 and 3)");
    return false;
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment(const Eigen::Vector3d& p0,
                                                   const Eigen::Vector3d& p1,
                                                   const Eigen::Vector3d& q0,
                                                   const Eigen::Vector3d& q1,
                                                   std::size_t gdim)
{
  switch (gdim)
  {
  case 1:
    return collides_segment_segment_1d(p0[0], p1[0], q0[0], q1[0]);
  case 2:
    return collides_segment_segment_2d(p0, p1, q0, q1);
  case 3:
    return collides_segment_segment_3d(p0, p1, q0, q1);
  default:
    throw std::runtime_error(
        "Cannot call collides_segment_segment. Unknown "
        "dimension (only implemented for dimension 2 and 3)");
    return false;
  }
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_point(const Eigen::Vector3d& p0,
                                                  const Eigen::Vector3d& p1,
                                                  const Eigen::Vector3d& p2,
                                                  const Eigen::Vector3d& point,
                                                  std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return collides_triangle_point_2d(p0, p1, p2, point);
  case 3:
    return collides_triangle_point_3d(p0, p1, p2, point);
  default:
    throw std::runtime_error(
        "Cannot call collides_triangle_point. Unknown "
        "dimension (only implemented for dimension 2 and 3)");
    return false;
  }
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_segment(const Eigen::Vector3d& p0,
                                                    const Eigen::Vector3d& p1,
                                                    const Eigen::Vector3d& p2,
                                                    const Eigen::Vector3d& q0,
                                                    const Eigen::Vector3d& q1,
                                                    std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return collides_triangle_segment_2d(p0, p1, p2, q0, q1);
  case 3:
    return collides_triangle_segment_3d(p0, p1, p2, q0, q1);
  default:
    throw std::runtime_error(
        "Cannot call collides_triangle_segment. Unknown "
        "dimension (only implemented for dimension 2 and 3)");
    return false;
  }
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_triangle(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& q0,
    const Eigen::Vector3d& q1, const Eigen::Vector3d& q2, std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return collides_triangle_triangle_2d(p0, p1, p2, q0, q1, q2);
  case 3:
    return collides_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);
  default:
    throw std::runtime_error(
        "Cannot call collides_triangle_triangle. Unknown "
        "dimension (only implemented for dimension 2 and 3)");
    return false;
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point_1d(double p0, double p1,
                                                    double point)
{
  if (p0 > p1)
    std::swap(p0, p1);
  return p0 <= point and point <= p1;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point_2d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& point)
{
  const double orientation = orient2d(p0, p1, point);
  const Eigen::Vector3d dp = p1 - p0;
  const double segment_length = dp.dot(dp);
  return orientation == 0.0 && (point - p0).dot(point - p0) <= segment_length
         && (point - p1).dot(point - p1) <= segment_length
         && dp.dot(p1 - point) >= 0.0 && dp.dot(point - p0) >= 0.0;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& point)
{
  if (point == p0 or point == p1)
    return true;

  // Poject to reduce to three 2d problems
  const double det_xy = orient2d(p0, p1, point);

  if (det_xy == 0.0)
  {
    const std::array<std::array<double, 2>, 3> xz
        = {{{{p0[0], p0[2]}}, {{p1[0], p1[2]}}, {{point[0], point[2]}}}};
    const double det_xz = _orient2d(xz[0].data(), xz[1].data(), xz[2].data());

    if (det_xz == 0.0)
    {
      const std::array<std::array<double, 2>, 3> yz
          = {{{{p0[1], p0[2]}}, {{p1[1], p1[2]}}, {{point[1], point[2]}}}};
      const double det_yz = _orient2d(yz[0].data(), yz[1].data(), yz[2].data());
      if (det_yz == 0.0)
      {
        // Point is aligned with segment
        const double length = (p0 - p1).dot(p0 - p1);
        return (point - p0).dot(point - p0) <= length
               and (point - p1).dot(point - p1) <= length;
      }
    }
  }

  return false;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment_1d(double p0, double p1,
                                                      double q0, double q1)
{
  // Get range
  const double a0 = std::min(p0, p1);
  const double b0 = std::max(p0, p1);
  const double a1 = std::min(q0, q1);
  const double b1 = std::max(q0, q1);

  // Check for collision
  const double dx = std::min(b0 - a0, b1 - a1);
  return b1 >= a0 - dx && a1 <= b0 + dx;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment_2d(const Eigen::Vector3d& p0,
                                                      const Eigen::Vector3d& p1,
                                                      const Eigen::Vector3d& q0,
                                                      const Eigen::Vector3d& q1)
{
  // FIXME: Optimize by avoiding redundant calls to orient2d

  if (collides_segment_point_2d(p0, p1, q0))
    return true;
  else if (collides_segment_point_2d(p0, p1, q1))
    return true;
  else if (collides_segment_point_2d(q0, q1, p0))
    return true;
  else if (collides_segment_point_2d(q0, q1, p1))
    return true;
  else
  {
    // Points must be on different sides
    if (((orient2d(q0, q1, p0) > 0.0) xor (orient2d(q0, q1, p1) > 0.0))
        and ((orient2d(p0, p1, q0) > 0.0) xor (orient2d(p0, p1, q1) > 0.0)))
    {
      return true;
    }
    else
      return false;
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment_3d(const Eigen::Vector3d& p0,
                                                      const Eigen::Vector3d& p1,
                                                      const Eigen::Vector3d& q0,
                                                      const Eigen::Vector3d& q1)
{
  // Vertex collisions
  if (p0 == q0 || p0 == q1 || p1 == q0 || p1 == q1)
    return true;

  if (collides_segment_point_3d(p0, p1, q0)
      or collides_segment_point_3d(p0, p1, q1)
      or collides_segment_point_3d(q0, q1, p0)
      or collides_segment_point_3d(q0, q1, p1))
  {
    return true;
  }

  // Determinant must be zero
  const double det = orient3d(p0, p1, q0, q1);

  if (det < 0.0 or det > 0.0)
    return false;

  // Now we know that the segments are in the same plane. This means
  // that they can be parallel, or even collinear.

  // Check for collinearity
  const Eigen::Vector3d u = cross_product(p0, p1, q0);
  if (u[0] == 0.0 and u[1] == 0.0 and u[2] == 0.0)
  {
    const Eigen::Vector3d v = cross_product(p0, p1, q1);
    if (v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0)
    {
      // Now we know that the segments are collinear
      if ((p0 - q0).dot(p0 - q0) <= (q1 - q0).dot(q1 - q0)
          and (p0 - q1).dot(p0 - q1) <= (q0 - q1).dot(q0 - q1))
      {
        return true;
      }

      if ((p1 - q0).dot(p1 - q0) <= (q1 - q0).dot(q1 - q0)
          and (p1 - q1).dot(p1 - q1) <= (q0 - q1).dot(q0 - q1))
      {
        return true;
      }

      if ((q0 - p0).dot(q0 - p0) <= (p1 - p0).dot(p1 - p0)
          and (q0 - p1).dot(q0 - p1) <= (p0 - p1).dot(p0 - p1))
      {
        return true;
      }

      if ((q1 - p0).dot(q1 - p0) <= (p1 - p0).dot(p1 - p0)
          and (q1 - p1).dot(q1 - p1) <= (p0 - p1).dot(p0 - p1))
      {
        return true;
      }
    }
  }

  // Segments are not collinear, but in the same plane
  // Try to reduce to 2d by elimination
  for (int d = 0; d < 3; ++d)
  {
    if (p0[d] == p1[d] and p0[d] == q0[d] and p0[d] == q1[d])
    {
      const std::array<std::array<std::size_t, 2>, 3> dims
          = {{{{1, 2}}, {{0, 2}}, {{0, 1}}}};
      Eigen::Vector3d p0_2d(p0[dims[d][0]], p0[dims[d][1]], 0.0);
      Eigen::Vector3d p1_2d(p1[dims[d][0]], p1[dims[d][1]], 0.0);
      Eigen::Vector3d q0_2d(q0[dims[d][0]], q0[dims[d][1]], 0.0);
      Eigen::Vector3d q1_2d(q1[dims[d][0]], q1[dims[d][1]], 0.0);

      return collides_segment_segment_2d(p0_2d, p1_2d, q0_2d, q1_2d);
    }
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_point_2d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& point)
{
  const double ref = orient2d(p0, p1, p2);

  if (ref > 0.0)
  {
    return (orient2d(p1, p2, point) >= 0.0 and orient2d(p2, p0, point) >= 0.0
            and orient2d(p0, p1, point) >= 0.0);
  }
  else if (ref < 0.0)
  {
    return (orient2d(p1, p2, point) <= 0.0 and orient2d(p2, p0, point) <= 0.0
            and orient2d(p0, p1, point) <= 0.0);
  }
  else
  {
    return ((orient2d(p0, p1, point) == 0.0
             and collides_segment_point_1d(p0[0], p1[0], point[0])
             and collides_segment_point_1d(p0[1], p1[1], point[1]))
            or (orient2d(p1, p2, point) == 0.0
                and collides_segment_point_1d(p1[0], p2[0], point[0])
                and collides_segment_point_1d(p1[1], p2[1], point[1]))
            or (orient2d(p2, p0, point) == 0.0
                and collides_segment_point_1d(p2[0], p0[0], point[0])
                and collides_segment_point_1d(p2[1], p0[1], point[1])));
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_quad_point_2d(const Eigen::Vector3d& p0,
                                                 const Eigen::Vector3d& p1,
                                                 const Eigen::Vector3d& p2,
                                                 const Eigen::Vector3d& p3,
                                                 const Eigen::Vector3d& point)
{
  const double ref0 = orient2d(p0, p1, p2);
  const double ref1 = orient2d(p3, p2, p1);

  if (ref0 * ref1 <= 0.0)
    throw std::runtime_error("Badly formed quadrilateral");

  if (ref0 > 0.0)
  {
    return (orient2d(p1, p3, point) >= 0.0 and orient2d(p3, p2, point) >= 0.0
            and orient2d(p2, p0, point) >= 0.0
            and orient2d(p0, p1, point) >= 0.0);
  }
  else
  {
    return (orient2d(p1, p3, point) <= 0.0 and orient2d(p3, p2, point) <= 0.0
            and orient2d(p2, p0, point) <= 0.0
            and orient2d(p0, p1, point) <= 0.0);
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_point_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& point)
{
  if (p0 == point or p1 == point or p2 == point)
    return true;

  const double tet_det = orient3d(p0, p1, p2, point);
  if (tet_det < 0.0 or tet_det > 0.0)
    return false;

  // Use normal
  const Eigen::Vector3d n = cross_product(p0, p1, p2);
  return !(n.dot(cross_product(point, p0, p1)) < 0.0
           or n.dot(cross_product(point, p2, p0)) < 0.0
           or n.dot(cross_product(point, p1, p2)) < 0.0);
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_segment_2d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& q0,
    const Eigen::Vector3d& q1)
{
  // FIXME: Optimize by avoiding redundant calls to orient2d

  // Check if end points are in triangle
  if (collides_triangle_point_2d(p0, p1, p2, q0))
    return true;
  if (collides_triangle_point_2d(p0, p1, p2, q1))
    return true;

  // Check if any of the triangle edges are cut by the segment
  if (collides_segment_segment_2d(p0, p1, q0, q1))
    return true;
  if (collides_segment_segment_2d(p0, p2, q0, q1))
    return true;
  if (collides_segment_segment_2d(p1, p2, q0, q1))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_segment_3d(const Eigen::Vector3d& r,
                                                       const Eigen::Vector3d& s,
                                                       const Eigen::Vector3d& t,
                                                       const Eigen::Vector3d& a,
                                                       const Eigen::Vector3d& b)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Compute correspondic tetrahedra determinants
  const double rsta = orient3d(r, s, t, a);
  const double rstb = orient3d(r, s, t, b);

  // Check if a and b are on same side of triangle rst
  if ((rsta < 0.0 and rstb < 0.0) or (rsta > 0.0 and rstb > 0.0))
    return false;

  // We check triangle point first. We use this below.
  if (collides_triangle_point_3d(r, s, t, a))
    return true;

  if (collides_triangle_point_3d(r, s, t, b))
    return true;

  // Now we know a and b are either on different sides or in the same
  // plane (in which case rsta = rstb = 0). Check if intersection is
  // in triangle by creating some other tets.

  if (rsta == 0.0 and rstb == 0.0)
  {
    // Since we have checked that the points does not collide, the
    // segment is either completely outside the triangle, or we have a
    // collision over edges.

    // FIXME: To avoid collision over edges, maybe we can test if both
    // a and b are on the same side of one of the edges rs, rt or st.

    if (collides_segment_segment_3d(r, s, a, b))
      return true;
    if (collides_segment_segment_3d(r, t, a, b))
      return true;
    if (collides_segment_segment_3d(s, t, a, b))
      return true;

    return false;
  }
  else
  {
    // Temporarily flip a and b to make sure a is above
    Eigen::Vector3d _a = a;
    Eigen::Vector3d _b = b;
    if (rsta < 0.0)
      std::swap(_a, _b);

    const double rasb = orient3d(r, _a, s, _b);
    if (rasb < 0)
      return false;

    const double satb = orient3d(s, _a, t, _b);
    if (satb < 0)
      return false;

    const double tarb = orient3d(t, _a, r, _b);
    if (tarb < 0)
      return false;
  }

  return true;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_triangle_2d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& q0,
    const Eigen::Vector3d& q1, const Eigen::Vector3d& q2)
{
  // FIXME: Optimize by avoiding redundant calls to orient2d

  // Pack points as vectors
  const std::array<Eigen::Vector3d, 3> tri_0 = {{p0, p1, p2}};
  const std::array<Eigen::Vector3d, 3> tri_1 = {{q0, q1, q2}};

  const bool s0 = std::signbit(orient2d(p0, p1, p2));
  const bool s1 = std::signbit(orient2d(q0, q1, q2));

  for (std::size_t i = 0; i < 3; ++i)
  {
    if ((s0 and orient2d(tri_0[0], tri_0[1], tri_1[i]) <= 0.0
         and orient2d(tri_0[1], tri_0[2], tri_1[i]) <= 0.0
         and orient2d(tri_0[2], tri_0[0], tri_1[i]) <= 0.0)
        or (!s0 and orient2d(tri_0[0], tri_0[1], tri_1[i]) >= 0.0
            and orient2d(tri_0[1], tri_0[2], tri_1[i]) >= 0.0
            and orient2d(tri_0[2], tri_0[0], tri_1[i]) >= 0.0))
    {
      return true;
    }

    if ((s1 and orient2d(tri_1[0], tri_1[1], tri_0[i]) <= 0.0
         and orient2d(tri_1[1], tri_1[2], tri_0[i]) <= 0.0
         and orient2d(tri_1[2], tri_1[0], tri_0[i]) <= 0.0)
        or (!s1 and orient2d(tri_1[0], tri_1[1], tri_0[i]) >= 0.0
            and orient2d(tri_1[1], tri_1[2], tri_0[i]) >= 0.0
            and orient2d(tri_1[2], tri_1[0], tri_0[i]) >= 0.0))
    {
      return true;
    }
  }

  // Find all edge-edge collisions
  for (std::size_t i0 = 0; i0 < 3; i0++)
  {
    const std::size_t j0 = (i0 + 1) % 3;
    const Eigen::Vector3d& p0 = tri_0[i0];
    const Eigen::Vector3d& q0 = tri_0[j0];
    for (std::size_t i1 = 0; i1 < 3; i1++)
    {
      const std::size_t j1 = (i1 + 1) % 3;
      const Eigen::Vector3d& p1 = tri_1[i1];
      const Eigen::Vector3d& q1 = tri_1[j1];
      if (collides_segment_segment_2d(p0, q0, p1, q1))
        return true;
    }
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_triangle_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& q0,
    const Eigen::Vector3d& q1, const Eigen::Vector3d& q2)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Pack points as vectors
  const std::array<Eigen::Vector3d, 3> tri_0 = {{p0, p1, p2}};
  const std::array<Eigen::Vector3d, 3> tri_1 = {{q0, q1, q2}};

  // First test edge-face collisions
  for (std::size_t i = 0; i < 3; ++i)
  {
    const std::size_t j = (i + 1) % 3;

    if (collides_triangle_segment_3d(p0, p1, p2, tri_1[i], tri_1[j]))
      return true;

    if (collides_triangle_segment_3d(q0, q1, q2, tri_0[i], tri_0[j]))
      return true;
  }

  // Test edge-edge collisions
  for (std::size_t i0 = 0; i0 < 3; i0++)
  {
    const std::size_t j0 = (i0 + 1) % 3;
    for (std::size_t i1 = 0; i1 < 3; i1++)
    {
      const std::size_t j1 = (i1 + 1) % 3;
      if (collides_segment_segment_3d(tri_0[i0], tri_0[j0], tri_1[i1],
                                      tri_1[j1]))
      {
        return true;
      }
    }
  }

  // FIXME
  // Test point-face collisions (could also be detected by
  // triangle_segment collision above)
  for (std::size_t i = 0; i < 3; ++i)
  {
    if (collides_triangle_point_3d(p0, p1, p2, tri_1[i]))
      return true;

    if (collides_triangle_point_3d(q0, q1, q2, tri_0[i]))
      return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_point_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
    const Eigen::Vector3d& point)
{
  const double ref = orient3d(p0, p1, p2, p3);

  if (ref > 0.0)
  {
    return (orient3d(p0, p1, p2, point) >= 0.0
            and orient3d(p0, p3, p1, point) >= 0.0
            and orient3d(p0, p2, p3, point) >= 0.0
            and orient3d(p1, p3, p2, point) >= 0.0);
  }
  else if (ref < 0.0)
  {
    return (orient3d(p0, p1, p2, point) <= 0.0
            and orient3d(p0, p3, p1, point) <= 0.0
            and orient3d(p0, p2, p3, point) <= 0.0
            and orient3d(p1, p3, p2, point) <= 0.0);
  }
  else
  {
    throw std::runtime_error("Cannot compute tetrahedron point collision. "
                             "Not implemented for degenerate tetrahedron");
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_segment_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
    const Eigen::Vector3d& q0, const Eigen::Vector3d& q1)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Segment vertex in tetrahedron collision
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;

  // Triangle-segment collision tests
  if (collides_triangle_segment_3d(p1, p2, p3, q0, q1))
    return true;
  if (collides_triangle_segment_3d(p0, p2, p3, q0, q1))
    return true;
  if (collides_triangle_segment_3d(p0, p1, p3, q0, q1))
    return true;
  if (collides_triangle_segment_3d(p0, p1, p2, q0, q1))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_triangle_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
    const Eigen::Vector3d& q0, const Eigen::Vector3d& q1,
    const Eigen::Vector3d& q2)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Triangle vertex in tetrahedron collision
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q2))
    return true;

  // Triangle-triangle collision tests
  if (collides_triangle_triangle_3d(q0, q1, q2, p1, p2, p3))
    return true;
  if (collides_triangle_triangle_3d(q0, q1, q2, p0, p2, p3))
    return true;
  if (collides_triangle_triangle_3d(q0, q1, q2, p0, p1, p3))
    return true;
  if (collides_triangle_triangle_3d(q0, q1, q2, p0, p1, p2))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_tetrahedron_3d(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& p3,
    const Eigen::Vector3d& q0, const Eigen::Vector3d& q1,
    const Eigen::Vector3d& q2, const Eigen::Vector3d& q3)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  const std::array<Eigen::Vector3d, 4> tetp = {{p0, p1, p2, p3}};
  const std::array<Eigen::Vector3d, 4> tetq = {{q0, q1, q2, q3}};

  // Triangle face collisions
  const std::array<std::array<std::size_t, 3>, 4> faces
      = {{{{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}}}};
  for (std::size_t i = 0; i < 4; ++i)
  {
    for (std::size_t j = 0; j < 4; ++j)
    {
      if (collides_triangle_triangle_3d(tetp[faces[i][0]], tetp[faces[i][1]],
                                        tetp[faces[i][2]], tetq[faces[j][0]],
                                        tetq[faces[j][1]], tetq[faces[j][2]]))
      {
        return true;
      }
    }
  }

  // Vertex in tetrahedron collision
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q2))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q3))
    return true;

  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p0))
    return true;
  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p1))
    return true;
  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p2))
    return true;
  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p3))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
