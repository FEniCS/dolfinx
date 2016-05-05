// Copyright (C) 2014-2016 Anders Logg and August Johansson
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
// Last changed: 2016-05-05

#include <dolfin/mesh/MeshEntity.h>
#include "IntersectionTriangulation.h"
#include "CGALExactArithmetic.h"
#include "CollisionDetection.h"
#include "predicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// High-level intersection triangulation functions
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate(const MeshEntity& entity_0,
                                       const MeshEntity& entity_1)
{
  // Get data
  const MeshGeometry& g0 = entity_0.mesh().geometry();
  const MeshGeometry& g1 = entity_1.mesh().geometry();
  const unsigned int* v0 = entity_0.entities(0);
  const unsigned int* v1 = entity_1.entities(0);

  // Pack data as vectors of points
  std::vector<Point> points_0;
  std::vector<Point> points_1;
  for (std::size_t i = 0; i <= entity_0.dim(); i++)
    points_0.push_back(g0.point(v0[i]));
  for (std::size_t i = 0; i <= entity_1.dim(); i++)
    points_1.push_back(g1.point(v1[i]));

  // Only look at first entity to get geometric dimension
  std::size_t gdim = g0.dim();

  // Call common implementation
  return triangulate(points_0, points_1, gdim);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate(const std::vector<Point>& points_0,
                                       const std::vector<Point>& points_1,
                                       std::size_t gdim)
{
  // Get topological dimensions
  const std::size_t d0 = points_0.size() - 1;
  const std::size_t d1 = points_1.size() - 1;

  // Pick correct specialized implementation
  if (d0 == 2 && d1 == 1)
    return triangulate_triangle_segment(points_0[0],
                                        points_0[1],
                                        points_0[2],
                                        points_1[0],
                                        points_1[1],
                                        gdim);

  if (d0 == 2 && d1 == 2)
    return triangulate_triangle_triangle(points_0[0],
                                         points_0[1],
                                         points_0[2],
                                         points_1[0],
                                         points_1[1],
                                         points_1[2]);

  if (d0 == 2 && d1 == 3)
    return triangulate_tetrahedron_triangle(points_1[0],
                                            points_1[1],
                                            points_1[2],
                                            points_1[3],
                                            points_0[0],
                                            points_0[1],
                                            points_0[2]);

  if (d0 == 3 && d1 == 2)
    return triangulate_tetrahedron_triangle(points_0[0],
                                            points_0[1],
                                            points_0[2],
                                            points_0[3],
                                            points_1[0],
                                            points_1[1],
                                            points_1[2]);

  if (d0 == 2 && d1 == 2)
    return triangulate_tetrahedron_tetrahedron(points_0[0],
                                               points_0[1],
                                               points_0[2],
                                               points_0[3],
                                               points_1[0],
                                               points_1[1],
                                               points_1[2],
                                               points_1[3]);

  dolfin_error("IntersectionTriangulation.cpp",
               "compute intersection triangulation",
               "Not implemented for dimensions %d / %d", d0, d1);

  return std::vector<double>();
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate(const MeshEntity &entity,
                                       const std::vector<double> &triangulation,
                                       std::size_t tdim)
{
  // Compute the triangulation of the intersection of the cell and the
  // simplices of the flat triangulation vector with topology tdim.

  std::vector<double> total_triangulation;

  // Get dimensions (geometrical dimension assumed to be the same)
  const std::size_t cell_tdim = entity.mesh().topology().dim();
  const std::size_t gdim = entity.mesh().geometry().dim();

  // Store cell as std::vector<Point>
  // FIXME: Store as Point& ?
  std::vector<Point> simplex_cell(cell_tdim + 1);
  const MeshGeometry& geometry = entity.mesh().geometry();
  const unsigned int* vertices = entity.entities(0);
  for (std::size_t j = 0; j < cell_tdim + 1; ++j)
    simplex_cell[j] = geometry.point(vertices[j]);

  // Simplex in triangulation
  std::vector<Point> simplex(tdim + 1);
  const std::size_t offset = (tdim + 1)*gdim;

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size() / offset; ++i)
  {
    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < tdim + 1; ++j)
      for (std::size_t d = 0; d < gdim; ++d)
        simplex[j][d] = triangulation[offset*i + gdim*j + d];

    // Compute intersection
    std::vector<double> local_triangulation
      = triangulate(simplex_cell, simplex, gdim);

    // Add these to the net triangulation
    total_triangulation.insert(total_triangulation.end(),
                               local_triangulation.begin(),
                               local_triangulation.end());
  }

  return total_triangulation;
}
//-----------------------------------------------------------------------------
void
IntersectionTriangulation::triangulate(const MeshEntity &entity,
                                       const std::vector<double>& triangulation,
                                       const std::vector<Point>& normals,
                                       std::vector<double>& intersection_triangulation,
                                       std::vector<Point>& intersection_normals,
                                       std::size_t tdim)
{
  // Compute the triangulation of the intersection of the cell and the
  // simplices of the flat triangulation vector with topology tdim.

  // FIXME: clear or not?
  // intersection_triangulation.clear();
  // intersection_normals.clear();

  // Get dimensions (geometrical dimension assumed to be the same)
  const std::size_t entity_tdim = entity.mesh().topology().dim();
  const std::size_t gdim = entity.mesh().geometry().dim();

  // Store entity as std::vector<Point>
  // FIXME: Store as Point& ?
  std::vector<Point> simplex_entity(entity_tdim+1);
  const MeshGeometry& geometry = entity.mesh().geometry();
  const unsigned int* vertices = entity.entities(0);
  for (std::size_t j = 0; j < entity_tdim + 1; ++j)
    simplex_entity[j] = geometry.point(vertices[j]);

  // Simplex in triangulation
  std::vector<Point> simplex(tdim + 1);
  const std::size_t offset = (tdim + 1)*gdim;

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size()/offset; ++i)
  {
    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < tdim + 1; ++j)
      for (std::size_t d = 0; d < gdim; ++d)
        simplex[j][d] = triangulation[offset*i + gdim*j + d];

    // Compute intersection
    std::vector<double> local_triangulation
      = triangulate(simplex_entity, simplex, gdim);

    // Add these to the net triangulation
    intersection_triangulation.insert(intersection_triangulation.end(),
                                      local_triangulation.begin(),
                                      local_triangulation.end());

    // Add the normal
    intersection_normals.resize(intersection_normals.size() + local_triangulation.size()/offset,
                                normals[i]);
  }
}
//-----------------------------------------------------------------------------
// Low-level intersection triangulation functions
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::_triangulate_segment_segment(const Point& p0,
							const Point& p1,
							const Point& q0,
							const Point& q1,
							std::size_t gdim)
{
  // Flat array for triangulation
  std::vector<double> triangulation;

  if (CollisionDetection::collides_segment_segment(p0, p1,
                                                   q0, q1))
  {
    // Compute collisions
    std::vector<Point> points;
    if (CollisionDetection::collides_segment_point(p0, p1, q0))
      points.push_back(q0);
    if (CollisionDetection::collides_segment_point(p0, p1, q1))
      points.push_back(q1);
    if (CollisionDetection::collides_segment_point(q0, q1, p0))
      points.push_back(p0);
    if (CollisionDetection::collides_segment_point(q0, q1, p0))
      points.push_back(p0);

    // Must not have more than two points
    if (points.size() == 2)
    {
      triangulation.resize(2*gdim);
      for (std::size_t d = 0; d < gdim; ++d)
      {
        triangulation[d] = points[0][d];
        triangulation[gdim+d] = points[1][d];
      }
    }
  }
  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::_triangulate_triangle_segment(const Point& p0,
							 const Point& p1,
							 const Point& p2,
							 const Point& q0,
							 const Point& q1,
							 std::size_t gdim)
{
  std::vector<double> triangulation;
  std::vector<Point> points;

  // Detect edge intersection points
  if (CollisionDetection::collides_segment_segment(p0, p1, q0, q1))
    points.push_back(_intersection_edge_edge(p0, p1, q0, q1));
  if (CollisionDetection::collides_segment_segment(p0, p2, q0, q1))
    points.push_back(_intersection_edge_edge(p0, p2, q0, q1));
  if (CollisionDetection::collides_segment_segment(p1, p2, q0, q1))
    points.push_back(_intersection_edge_edge(p1, p2, q0, q1));

  // If we get zero intersection points, then both segment ends must
  // be inside
  // FIXME: can we really use two different types of intersection tests: intersection_edge_edge above and Collides here?
  if (points.size() == 0)
  {
    if (CollisionDetection::collides_triangle_point(p0, p1, p2, q0) and
        CollisionDetection::collides_triangle_point(p0, p1, p2, q1))
    {
      triangulation.resize(2*gdim);
      for (std::size_t d = 0; d < gdim; ++d)
      {
        triangulation[d] = q0[d];
        triangulation[gdim + d] = q1[d];
      }
      return triangulation;
    }
  }

  // If we get one intersection point, find the segment end point
  // which is inside the triangle. Note that this point should
  // absolutely not be the same point as we found above. This can
  // happen since we use different types of tests here and above.
  if (points.size() == 1)
  {
    // Make sure the point q0 is not points[0]
    if ((q0 - points[0]).norm() > DOLFIN_EPS_LARGE and
        CollisionDetection::collides_triangle_point(p0, p1, p2, q0))
    {
      triangulation.resize(2*gdim);
      for (std::size_t d = 0; d < gdim; ++d)
      {
        triangulation[d] = points[0][d];
        triangulation[gdim+d] = q0[d];
      }
      return triangulation;
    }

    // Make sure the point q1 is not points[0]
    if ((q0 - points[0]).norm() > DOLFIN_EPS_LARGE and
        CollisionDetection::collides_triangle_point(p0, p1, p2, q0))
    {
      triangulation.resize(2*gdim);
      for (std::size_t d = 0; d < gdim; ++d)
      {
        triangulation[d] = points[0][d];
        triangulation[gdim+d] = q0[d];
      }
      return triangulation;
    }
  }

  // If we get two intersection points, triangulate this line.
  if (points.size() == 2)
  {
    triangulation.resize(2*gdim);
    for (std::size_t d = 0; d < gdim; ++d)
    {
      triangulation[d] = points[0][d];
      triangulation[gdim + d] = points[1][d];
    }
    return triangulation;
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::_triangulate_triangle_triangle(const Point& p0,
							  const Point& p1,
							  const Point& p2,
							  const Point& q0,
							  const Point& q1,
							  const Point& q2)
{
  // This algorithm computes the (convex) polygon resulting from the
  // intersection of two triangles. It then triangulates the polygon
  // by trivially drawing an edge from one vertex to all other
  // vertices. The polygon is computed by first identifying all
  // vertex-cell collisions and then all edge-edge collisions. The
  // points are then sorted using a simplified Graham scan (simplified
  // since we know the polygon is convex).

  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  //const double same_point_tol = DOLFIN_EPS_LARGE;

  // Pack points as vectors
  std::vector<Point> tri_0({p0, p1, p2});
  std::vector<Point> tri_1({q0, q1, q2});

  // Create empty list of collision points
  std::vector<Point> points;

  // Extract coordinates
  double t0[3][2] = {{p0[0], p0[1]}, {p1[0], p1[1]}, {p2[0], p2[1]}};
  double t1[3][2] = {{q0[0], q0[1]}, {q1[0], q1[1]}, {q2[0], q2[1]}};

  // Find all vertex-cell collisions
  const int s0 = std::signbit(orient2d(t0[0], t0[1], t0[2])) == true ? -1 : 1;
  const int s1 = std::signbit(orient2d(t1[0], t1[1], t1[2])) == true ? -1 : 1;

  for (std::size_t i = 0; i < 3; ++i)
  {
    if (s1*orient2d(t1[0], t1[1], t0[i]) >= 0. and
  	s1*orient2d(t1[1], t1[2], t0[i]) >= 0. and
  	s1*orient2d(t1[2], t1[0], t0[i]) >= 0.)
      points.push_back(tri_0[i]);

    if (s0*orient2d(t0[0], t0[1], t1[i]) >= 0. and
  	s0*orient2d(t0[1], t0[2], t1[i]) >= 0. and
  	s0*orient2d(t0[2], t0[0], t1[i]) >= 0.)
      points.push_back(tri_1[i]);
  }

  // Find all edge-edge collisions
  for (std::size_t i0 = 0; i0 < 3; i0++)
  {
    const std::size_t j0 = (i0 + 1) % 3;
    const Point& p0 = tri_0[i0];
    const Point& q0 = tri_0[j0];
    for (std::size_t i1 = 0; i1 < 3; i1++)
    {
      const std::size_t j1 = (i1 + 1) % 3;
      const Point& p1 = tri_1[i1];
      const Point& q1 = tri_1[j1];
      if (CollisionDetection::collides_segment_segment(p0, q0, p1, q1))
	points.push_back(_intersection_edge_edge_2d(p0, q0, p1, q1));
    }
  }

  // Remove duplicate points
  std::vector<Point> tmp;
  tmp.reserve(points.size());

  for (std::size_t i = 0; i < points.size(); ++i)
  {
    bool different = true;
    for (std::size_t j = i+1; j < points.size(); ++j)
      if ((points[i] - points[j]).norm() < DOLFIN_EPS)//_LARGE)
      {
  	different = false;
  	break;
      }
    if (different)
      tmp.push_back(points[i]);
  }
  points = tmp;

  // Special case: no points found
  std::vector<double> triangulation;
  if (points.size() < 3)
    return triangulation;

  // If the number of points are three, then these form the triangulation
  if (points.size() == 3)
  {
    triangulation.resize(6);

    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 2; ++j)
  	triangulation[2*i+j] = points[i][j];

    return triangulation;
  }

  // Find left-most point (smallest x-coordinate)
  std::size_t i_min = 0;
  double x_min = points[0].x();
  for (std::size_t i = 1; i < points.size(); i++)
  {
    const double x = points[i].x();
    if (x < x_min)
    {
      x_min = x;
      i_min = i;
    }
  }

  // Compute signed squared cos of angle with (0, 1) from i_min to all points
  std::vector<std::pair<double, std::size_t>> order;
  for (std::size_t i = 0; i < points.size(); i++)
  {
    // Skip left-most point used as origin
    if (i == i_min)
      continue;

    // Compute vector to point
    const Point v = points[i] - points[i_min];

    // Compute square cos of angle
    const double cos2 = (v.y() < 0.0 ? -1.0 : 1.0)*v.y()*v.y() / v.squared_norm();

    // Store for sorting
    order.push_back(std::make_pair(cos2, i));
  }

  // Sort points based on angle
  std::sort(order.begin(), order.end());

  // Triangulate polygon by connecting i_min with the ordered points
  triangulation.reserve((points.size() - 2)*3*2);
  const Point& _p0 = points[i_min];
  for (std::size_t i = 0; i < points.size() - 2; i++)
  {
    const Point& _p1 = points[order[i].second];
    const Point& _p2 = points[order[i + 1].second];
    triangulation.push_back(_p0.x());
    triangulation.push_back(_p0.y());
    triangulation.push_back(_p1.x());
    triangulation.push_back(_p1.y());
    triangulation.push_back(_p2.x());
    triangulation.push_back(_p2.y());
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::_triangulate_tetrahedron_triangle(const Point& p0,
							     const Point& p1,
							     const Point& p2,
							     const Point& p3,
							     const Point& q0,
							     const Point& q1,
							     const Point& q2)
{
  // This code mimics the
  // triangulate_tetrahedron_tetrahedron and the
  // triangulate_tetrahedron_tetrahedron_triangle_codes:
  // we first identify triangle nodes in the tetrahedra. Them we
  // continue with edge-face detection for the four faces of the
  // tetrahedron and the triangle. The points found are used to form a
  // triangulation by first sorting them using a Graham scan.

  // Pack points as vectors
  std::vector<Point> tet({p0, p1, p2, p3});
  std::vector<Point> tri({q0, q1, q2});

  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  const double same_point_tol = DOLFIN_EPS_LARGE;

  // Tolerance for small triangle (could be improved by identifying
  // sliver and small triangles)
  const double tri_det_tol = DOLFIN_EPS_LARGE;

  std::vector<Point> points;

  // Triangle node in tetrahedron intersection
  for (std::size_t i = 0; i < 3; ++i)
    if (CollisionDetection::collides_tetrahedron_point(tet[0],
                                                       tet[1],
                                                       tet[2],
                                                       tet[3],
                                                       tri[i]))
      points.push_back(tri[i]);

  // Check if a tetrahedron edge intersects the triangle
  std::vector<std::vector<int>> tet_edges(6, std::vector<int>(2));
  tet_edges[0][0] = 2;
  tet_edges[0][1] = 3;
  tet_edges[1][0] = 1;
  tet_edges[1][1] = 3;
  tet_edges[2][0] = 1;
  tet_edges[2][1] = 2;
  tet_edges[3][0] = 0;
  tet_edges[3][1] = 3;
  tet_edges[4][0] = 0;
  tet_edges[4][1] = 2;
  tet_edges[5][0] = 0;
  tet_edges[5][1] = 1;

  for (std::size_t e = 0; e < 6; ++e)
    if (CollisionDetection::collides_triangle_segment(tri[0], tri[1], tri[2],
                                                      tet[tet_edges[e][0]],
                                                      tet[tet_edges[e][1]]))
      points.push_back(_intersection_face_edge(tri[0], tri[1], tri[2],
                                               tet[tet_edges[e][0]],
                                               tet[tet_edges[e][1]]));

  // check if a triangle edge intersects a tetrahedron face
  std::vector<std::vector<std::size_t>>
    tet_faces(4, std::vector<std::size_t>(3));

  tet_faces[0][0] = 1;
  tet_faces[0][1] = 2;
  tet_faces[0][2] = 3;
  tet_faces[1][0] = 0;
  tet_faces[1][1] = 2;
  tet_faces[1][2] = 3;
  tet_faces[2][0] = 0;
  tet_faces[2][1] = 1;
  tet_faces[2][2] = 3;
  tet_faces[3][0] = 0;
  tet_faces[3][1] = 1;
  tet_faces[3][2] = 2;

  for (std::size_t f = 0; f < 4; ++f)
  {
    if (CollisionDetection::collides_triangle_segment(tet[tet_faces[f][0]],
						      tet[tet_faces[f][1]],
						      tet[tet_faces[f][2]],
						      tri[0], tri[1]))
      points.push_back(_intersection_face_edge(tet[tet_faces[f][0]],
					       tet[tet_faces[f][1]],
					       tet[tet_faces[f][2]],
					       tri[0], tri[1]));

    if (CollisionDetection::collides_triangle_segment(tet[tet_faces[f][0]],
                                                      tet[tet_faces[f][1]],
                                                      tet[tet_faces[f][2]],
                                                      tri[0], tri[2]))
      points.push_back(_intersection_face_edge(tet[tet_faces[f][0]],
                                               tet[tet_faces[f][1]],
                                               tet[tet_faces[f][2]],
                                               tri[0], tri[2]));

    if (CollisionDetection::collides_triangle_segment(tet[tet_faces[f][0]],
                                                      tet[tet_faces[f][1]],
                                                      tet[tet_faces[f][2]],
                                                      tri[1], tri[2]))
      points.push_back(_intersection_face_edge(tet[tet_faces[f][0]],
					       tet[tet_faces[f][1]],
					       tet[tet_faces[f][2]],
					       tri[1], tri[2]));
  }

  // edge edge intersection
  for (std::size_t f = 0; f < 6; ++f)
  {
    if (CollisionDetection::collides_segment_segment(tet[tet_edges[f][0]],
                                                     tet[tet_edges[f][1]],
                                                     tri[0], tri[1]))
      points.push_back(_intersection_edge_edge(tet[tet_edges[f][0]],
                                               tet[tet_edges[f][1]],
                                               tri[0], tri[1]));

    if (CollisionDetection::collides_segment_segment(tet[tet_edges[f][0]],
                                                     tet[tet_edges[f][1]],
                                                     tri[0], tri[2]))
      points.push_back(_intersection_edge_edge(tet[tet_edges[f][0]],
                                               tet[tet_edges[f][1]],
                                               tri[0], tri[2]));

    if (CollisionDetection::collides_segment_segment(tet[tet_edges[f][0]],
                                                     tet[tet_edges[f][1]],
                                                     tri[1], tri[2]))
      points.push_back(_intersection_edge_edge(tet[tet_edges[f][0]],
                                               tet[tet_edges[f][1]],
                                               tri[1], tri[2]));
  }


  // Remove duplicate nodes
  std::vector<Point> tmp;
  tmp.reserve(points.size());

  for (std::size_t i = 0; i < points.size(); ++i)
  {
    bool different = true;
    for (std::size_t j = i+1; j < points.size(); ++j)
      if ((points[i] - points[j]).norm() < same_point_tol)
      {
	different = false;
	break;
      }
    if (different)
      tmp.push_back(points[i]);
  }
  points = tmp;

  // We didn't find sufficiently many points
  if (points.size() < 3)
    return std::vector<double>();

  std::vector<double> triangulation;

  Point n = (points[2] - points[0]).cross(points[1] - points[0]);
  const double det = n.norm();
  n /= det;

  if (points.size() == 3) {
    // Include if determinant is sufficiently large
    if (det > tri_det_tol)
    {
      // One triangle with three vertices in 3D gives 9 doubles
      triangulation.resize(9);
      for (std::size_t m = 0, idx = 0; m < 3; ++m)
	for (std::size_t d = 0; d < 3; ++d, ++idx)
	  triangulation[idx] = points[m][d];
    }
    return triangulation;
  }

  // Tessellate as in the triangle-triangle intersection case: First
  // sort points using a Graham scan, then connect to form triangles.

  // Use the center of the points and point no 0 as reference for the
  // angle calculation
  Point pointscenter = points[0];
  for (std::size_t m = 1; m < points.size(); ++m)
    pointscenter += points[m];
  pointscenter /= points.size();

  std::vector<std::pair<double, std::size_t>> order;
  Point ref = points[0]-pointscenter;
  ref /= ref.norm();

  // Calculate and store angles
  for (std::size_t m = 1; m < points.size(); ++m)
  {
    const Point v = points[m] - pointscenter;
    const double frac = ref.dot(v) / v.norm();
    double alpha;
    if (frac <= -1)
      alpha = DOLFIN_PI;
    else if (frac >= 1)
      alpha = 0;
    else
    {
      alpha = acos(frac);
      if (v.dot(n.cross(ref)) < 0)
        alpha = 2*DOLFIN_PI-alpha;
    }
    order.push_back(std::make_pair(alpha, m));
  }

  // Sort angles
  std::sort(order.begin(), order.end());

  // Tessellate
  std::vector<Point> cand(3);
  for (std::size_t m = 0; m < order.size()-1; ++m)
  {
    // Candidate triangle
    cand[0] = points[0];
    cand[1] = points[order[m].second];
    cand[2] = points[order[m + 1].second];

    // Include triangle if determinant is sufficiently large
    const double det = ((cand[2] - cand[1]).cross(cand[1] - cand[0])).norm();
    if (det > tri_det_tol)
    {
      for (std::size_t n = 0; n < 3; ++n)
	for (std::size_t d = 0; d < 3; ++d)
	  triangulation.push_back(cand[n][d]);
    }
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::_triangulate_tetrahedron_tetrahedron(const Point& p0,
								const Point& p1,
								const Point& p2,
								const Point& p3,
								const Point& q0,
								const Point& q1,
								const Point& q2,
								const Point& q3)
{
  // This algorithm computes the intersection of cell_0 and cell_1 by
  // returning a vector<double> with points describing a tetrahedral
  // mesh of the intersection. We will use the fact that the
  // intersection is a convex polyhedron. The algorithm works by first
  // identifying intersection points: vertex points inside a cell,
  // edge-face collision points and edge-edge collision points (the
  // edge-edge is a rare occurance). Having the intersection points,
  // we identify points that are coplanar and thus form a facet of the
  // polyhedron. These points are then used to form a tessellation of
  // triangles, which are used to form tetrahedra by the use of the
  // center point of the polyhedron. This center point is thus an
  // additional point not found on the polyhedron facets.

  // Pack points as vectors
  std::vector<Point> tet_0({p0, p1, p2, p3});
  std::vector<Point> tet_1({q0, q1, q2, q3});

  // Tolerance for coplanar points
  const double coplanar_tol = 1000*DOLFIN_EPS_LARGE;

  // Tolerance for the tetrahedron determinant (otherwise problems
  // with warped tets)
  const double tet_det_tol = DOLFIN_EPS_LARGE;

  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  const double same_point_tol = DOLFIN_EPS_LARGE;

  // Tolerance for small triangle (could be improved by identifying
  // sliver and small triangles)
  const double tri_det_tol = DOLFIN_EPS_LARGE;

  // Points in the triangulation (unique)
  std::vector<Point> points;

  // Node intersection
  for (int i = 0; i < 4; ++i)
  {
    if (CollisionDetection::collides_tetrahedron_point(tet_0[0],
                                                       tet_0[1],
                                                       tet_0[2],
                                                       tet_0[3],
                                                       tet_1[i]))
      points.push_back(tet_1[i]);

    if (CollisionDetection::collides_tetrahedron_point(tet_1[0],
                                                       tet_1[1],
                                                       tet_1[2],
                                                       tet_1[3],
                                                       tet_0[i]))
      points.push_back(tet_0[i]);
  }

  // Edge face intersections
  std::vector<std::vector<std::size_t>> edges_0(6, std::vector<std::size_t>(2));
  edges_0[0][0] = 2;
  edges_0[0][1] = 3;
  edges_0[1][0] = 1;
  edges_0[1][1] = 3;
  edges_0[2][0] = 1;
  edges_0[2][1] = 2;
  edges_0[3][0] = 0;
  edges_0[3][1] = 3;
  edges_0[4][0] = 0;
  edges_0[4][1] = 2;
  edges_0[5][0] = 0;
  edges_0[5][1] = 1;

  std::vector<std::vector<std::size_t>> edges_1(6, std::vector<std::size_t>(2));
  edges_1[0][0] = 2;
  edges_1[0][1] = 3;
  edges_1[1][0] = 1;
  edges_1[1][1] = 3;
  edges_1[2][0] = 1;
  edges_1[2][1] = 2;
  edges_1[3][0] = 0;
  edges_1[3][1] = 3;
  edges_1[4][0] = 0;
  edges_1[4][1] = 2;
  edges_1[5][0] = 0;
  edges_1[5][1] = 1;

  std::vector<std::vector<std::size_t>> faces_0(4, std::vector<std::size_t>(3));
  faces_0[0][0] = 1;
  faces_0[0][1] = 2;
  faces_0[0][2] = 3;
  faces_0[1][0] = 0;
  faces_0[1][1] = 2;
  faces_0[1][2] = 3;
  faces_0[2][0] = 0;
  faces_0[2][1] = 1;
  faces_0[2][2] = 3;
  faces_0[3][0] = 0;
  faces_0[3][1] = 1;
  faces_0[3][2] = 2;

  std::vector<std::vector<std::size_t>> faces_1(4, std::vector<std::size_t>(3));
  faces_1[0][0] = 1;
  faces_1[0][1] = 2;
  faces_1[0][2] = 3;
  faces_1[1][0] = 0;
  faces_1[1][1] = 2;
  faces_1[1][2] = 3;
  faces_1[2][0] = 0;
  faces_1[2][1] = 1;
  faces_1[2][2] = 3;
  faces_1[3][0] = 0;
  faces_1[3][1] = 1;
  faces_1[3][2] = 2;

  // Loop over edges e and faces f
  for (std::size_t e = 0; e < 6; ++e)
  {
    for (std::size_t f = 0; f < 4; ++f)
    {
      if (CollisionDetection::collides_triangle_segment(tet_0[faces_0[f][0]],
                                                        tet_0[faces_0[f][1]],
                                                        tet_0[faces_0[f][2]],
                                                        tet_1[edges_1[e][0]],
                                                        tet_1[edges_1[e][1]]))
	points.push_back(_intersection_face_edge(tet_0[faces_0[f][0]],
                                                 tet_0[faces_0[f][1]],
                                                 tet_0[faces_0[f][2]],
                                                 tet_1[edges_1[e][0]],
                                                 tet_1[edges_1[e][1]]));

      if (CollisionDetection::collides_triangle_segment(tet_1[faces_1[f][0]],
                                                        tet_1[faces_1[f][1]],
                                                        tet_1[faces_1[f][2]],
                                                        tet_0[edges_0[e][0]],
                                                        tet_0[edges_0[e][1]]))
	points.push_back(_intersection_face_edge(tet_1[faces_1[f][0]],
                                                 tet_1[faces_1[f][1]],
                                                 tet_1[faces_1[f][2]],
                                                 tet_0[edges_0[e][0]],
                                                 tet_0[edges_0[e][1]]));
    }
  }

  // Edge edge intersection
  Point pt;
  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      if (CollisionDetection::collides_segment_segment(tet_0[edges_0[i][0]],
                                                       tet_0[edges_0[i][1]],
                                                       tet_1[edges_1[j][0]],
                                                       tet_1[edges_1[j][1]]))
	points.push_back(_intersection_edge_edge(tet_0[edges_0[i][0]],
                                                 tet_0[edges_0[i][1]],
                                                 tet_1[edges_1[j][0]],
                                                 tet_1[edges_1[j][1]]));
    }
  }

  // Remove duplicate nodes
  std::vector<Point> tmp;
  tmp.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    bool different=true;
    for (std::size_t j = i+1; j < points.size(); ++j)
    {
      if ((points[i] - points[j]).norm() < same_point_tol)
      {
  	different = false;
  	break;
      }
    }

    if (different)
      tmp.push_back(points[i]);
  }
  points = tmp;

  // We didn't find sufficiently many points: can't form any
  // tetrahedra.
  if (points.size() < 4)
    return std::vector<double>();

  // Points forming the tetrahedral partitioning of the polyhedron. We
  // have 4 points per tetrahedron in three dimensions => 12 doubles
  // per tetrahedron.
  std::vector<double> triangulation;

  // Start forming a tessellation
  if (points.size() == 4)
  {
    // Include if determinant is sufficiently large. The determinant
    // can possibly be computed in a more stable way if needed.
    const double det = (points[3] - points[0]).dot
      ((points[1] - points[0]).cross(points[2] - points[0]));

    if (std::abs(det) > tet_det_tol)
    {
      if (det < -tet_det_tol)
        std::swap(points[0], points[1]);

      // One tet with four vertices in 3D gives 12 doubles
      triangulation.resize(12);
      for (std::size_t m = 0, idx = 0; m < 4; ++m)
  	for (std::size_t d = 0; d < 3; ++d, ++idx)
  	  triangulation[idx] = points[m][d];
    }
    // Note: this can be empty if the tetrahedron was not sufficiently
    // large
    return triangulation;
  }

  // Tetrahedra are created using the facet points and a center point.
  Point polyhedroncenter = points[0];
  for (std::size_t i = 1; i < points.size(); ++i)
    polyhedroncenter += points[i];
  polyhedroncenter /= points.size();

  // Data structure for storing checked triangle indices (do this
  // better with some fancy stl structure?)
  const std::size_t N = points.size(), N2 = points.size()*points.size();
  std::vector<bool> checked(N*N2 + N2 + N, false);

  // Find coplanar points
  for (std::size_t i = 0; i < N; ++i)
  {
    for (std::size_t j = i+1; j < N; ++j)
    {
      for (std::size_t k = 0; k < N; ++k)
      {
  	if (!checked[i*N2 + j*N + k] and k != i and k != j)
  	{
  	  // Check that triangle area is sufficiently large
  	  Point n = (points[j] - points[i]).cross(points[k] - points[i]);
  	  const double tridet = n.norm();
  	  if (tridet < tri_det_tol)
            break;

  	  // Normalize normal
  	  n /= tridet;

  	  // Compute triangle center
  	  const Point tricenter = (points[i] + points[j] + points[k]) / 3.;

  	  // Check whether all other points are on one side of thus
  	  // facet. Initialize as true for the case of only three
  	  // coplanar points.
  	  bool on_convex_hull = true;

  	  // Compute dot products to check which side of the plane
  	  // (i,j,k) we're on. Note: it seems to be better to compute
  	  // n.dot(points[m]-n.dot(tricenter) rather than
  	  // n.dot(points[m]-tricenter).
  	  std::vector<double> ip(N, -(n.dot(tricenter)));
  	  for (std::size_t m = 0; m < N; ++m)
  	    ip[m] += n.dot(points[m]);

  	  // Check inner products range by finding max & min (this
  	  // seemed possibly more numerically stable than checking all
  	  // vs all and then break).
  	  double minip = 9e99, maxip = -9e99;
  	  for (size_t m = 0; m < N; ++m)
  	    if (m != i and m != j and m != k)
  	    {
  	      minip = (minip > ip[m]) ? ip[m] : minip;
  	      maxip = (maxip < ip[m]) ? ip[m] : maxip;
  	    }

  	  // Different sign => triangle is not on the convex hull
  	  if (minip*maxip < -DOLFIN_EPS)
  	    on_convex_hull = false;

  	  if (on_convex_hull)
  	  {
  	    // Find all coplanar points on this facet given the
  	    // tolerance coplanar_tol
  	    std::vector<std::size_t> coplanar;
  	    for (std::size_t m = 0; m < N; ++m)
  	      if (std::abs(ip[m]) < coplanar_tol)
  		coplanar.push_back(m);

  	    // Mark this plane (how to do this better?)
  	    for (std::size_t m = 0; m < coplanar.size(); ++m)
  	      for (std::size_t n = m+1; n < coplanar.size(); ++n)
  		for (std::size_t o = n+1; o < coplanar.size(); ++o)
  		  checked[coplanar[m]*N2 + coplanar[n]*N + coplanar[o]]
                    = checked[coplanar[m]*N2 + coplanar[o]*N + coplanar[n]]
  		    = checked[coplanar[n]*N2 + coplanar[m]*N + coplanar[o]]
  		    = checked[coplanar[n]*N2 + coplanar[o]*N + coplanar[m]]
  		    = checked[coplanar[o]*N2 + coplanar[n]*N + coplanar[m]]
  		    = checked[coplanar[o]*N2 + coplanar[m]*N + coplanar[n]]
                    = true;

  	    // Do the actual tessellation using the coplanar points and
  	    // a center point
  	    if (coplanar.size() == 3)
  	    {
  	      // Form one tetrahedron
  	      std::vector<Point> cand(4);
  	      cand[0] = points[coplanar[0]];
  	      cand[1] = points[coplanar[1]];
  	      cand[2] = points[coplanar[2]];
  	      cand[3] = polyhedroncenter;

  	      // Include if determinant is sufficiently large
  	      const double det = (cand[3]-cand[0]).dot
                ((cand[1] - cand[0]).cross(cand[2] - cand[0]));
  	      if (std::abs(det) > tet_det_tol)
  	      {
  		if (det < -tet_det_tol)
  		  std::swap(cand[0], cand[1]);

  		for (std::size_t m = 0; m < 4; ++m)
  		  for (std::size_t d = 0; d < 3; ++d)
  		    triangulation.push_back(cand[m][d]);
  	      }

  	    }
  	    else if (coplanar.size() > 3)
  	    {
  	      // Tessellate as in the triangle-triangle intersection
  	      // case: First sort points using a Graham scan, then
  	      // connect to form triangles. Finally form tetrahedra
  	      // using the center of the polyhedron.

  	      // Use the center of the coplanar points and point no 0
  	      // as reference for the angle calculation
  	      Point pointscenter = points[coplanar[0]];
  	      for (std::size_t m = 1; m < coplanar.size(); ++m)
  		pointscenter += points[coplanar[m]];
  	      pointscenter /= coplanar.size();

  	      std::vector<std::pair<double, std::size_t>> order;
  	      Point ref = points[coplanar[0]] - pointscenter;
  	      ref /= ref.norm();

  	      // Calculate and store angles
  	      for (std::size_t m = 1; m < coplanar.size(); ++m)
  	      {
  		const Point v = points[coplanar[m]] - pointscenter;
  		const double frac = ref.dot(v) / v.norm();
  		double alpha;
  		if (frac <= -1)
                  alpha=DOLFIN_PI;
  		else if (frac>=1)
                  alpha=0;
  		else
                {
  		  alpha = acos(frac);
  		  if (v.dot(n.cross(ref)) < 0)
                    alpha = 2*DOLFIN_PI-alpha;
  		}
  		order.push_back(std::make_pair(alpha, m));
  	      }

  	      // Sort angles
  	      std::sort(order.begin(), order.end());

  	      // Tessellate
  	      for (std::size_t m = 0; m < coplanar.size()-2; ++m)
  	      {
  		// Candidate tetrahedron:
  		std::vector<Point> cand(4);
  		cand[0] = points[coplanar[0]];
  		cand[1] = points[coplanar[order[m].second]];
  		cand[2] = points[coplanar[order[m + 1].second]];
  		cand[3] = polyhedroncenter;

  		// Include tetrahedron if determinant is "large"
  		const double det = (cand[3] - cand[0]).dot
                  ((cand[1] - cand[0]).cross(cand[2] - cand[0]));
  		if (std::abs(det) > tet_det_tol)
  		{
  		  if (det < -tet_det_tol)
  		    std::swap(cand[0], cand[1]);
  		  for (std::size_t n = 0; n < 4; ++n)
  		    for (std::size_t d = 0; d < 3; ++d)
  		      triangulation.push_back(cand[n][d]);
  		}
  	      }
  	    }
  	  }
  	}
      }
    }
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
// Private functions
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::graham_scan(const std::vector<Point>& points0)
{
  // Sometimes (at least using CGAL::intersection) we can get an extra
  // point on an edge: a-----c--b. This point c may cause problems for
  // the graham scan. To avoid this, use an extra center point.

  std::vector<double> triangulation;

#ifdef Augustdebug
  std::cout << "before duplicates "<< points0.size() << '\n';
  for (const auto p: points0)
    std::cout << tools::matlabplot(p);
  std::cout << '\n';
#endif

  // Remove duplicate points
  std::vector<Point> points;
  points.reserve(points0.size());

  for (std::size_t i = 0; i < points0.size(); ++i)
  {
    bool different = true;
    for (std::size_t j = i+1; j < points0.size(); ++j)
      if ((points0[i] - points0[j]).norm() < DOLFIN_EPS)
      {
	different = false;
	break;
      }
    if (different)
      points.push_back(points0[i]);
  }

  if (points.size() < 3)
  {
#ifdef Augustdebug
    std::cout << "after duplicate removal: " << points.size() << " too few points to form triangulation" << std::endl;
#endif
    return triangulation;
  }

  // Use the center of the points and point no 0 as reference for the
  // angle calculation
  Point pointscenter = points[0];
  for (std::size_t m = 1; m < points.size(); ++m)
    pointscenter += points[m];
  pointscenter /= points.size();

  Point n = (points[2] - points[0]).cross(points[1] - points[0]);
  const double det = n.norm();
  n /= det;

  std::vector<std::pair<double, std::size_t>> order;
  Point ref = points[0] - pointscenter;
  ref /= ref.norm();

  // Calculate and store angles
  for (std::size_t m = 1; m < points.size(); ++m)
  {
    const Point v = points[m] - pointscenter;
    const double frac = ref.dot(v) / v.norm();
    double alpha;
    if (frac <= -1)
      alpha = DOLFIN_PI;
    else if (frac >= 1)
      alpha = 0;
    else
    {
      alpha = acos(frac);
      if (v.dot(n.cross(ref)) < 0)
        alpha = 2*DOLFIN_PI-alpha;
    }
    order.push_back(std::make_pair(alpha, m));
  }

  // Sort angles
  std::sort(order.begin(), order.end());

#ifdef Augustdebug
  std::cout <<"plot("<<points[0].x()<<","<<points[0].y()<<");\n";
  for (const auto o: order)
    std::cout << "plot("<<points[o.second].x()<<','<<points[o.second].y()<<");\n";
#endif

  // Triangulate polygon by connecting i_min with the ordered points
  triangulation.reserve((points.size() - 2)*3*2);
  const Point& p0 = points[0];
  for (std::size_t i = 0; i < order.size() - 1; i++)
  {
    const Point& p1 = points[order[i].second];
    const Point& p2 = points[order[i + 1].second];
    triangulation.push_back(p0.x());
    triangulation.push_back(p0.y());
    triangulation.push_back(p1.x());
    triangulation.push_back(p1.y());
    triangulation.push_back(p2.x());
    triangulation.push_back(p2.y());
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
Point IntersectionTriangulation::_intersection_edge_edge_2d(const Point& a,
                                                            const Point& b,
                                                            const Point& c,
                                                            const Point& d)
{
  // Test Shewchuk style
  const double cda = orient2d(const_cast<double*>(c.coordinates()),
  			      const_cast<double*>(d.coordinates()),
  			      const_cast<double*>(a.coordinates()));
  const double cdb = orient2d(const_cast<double*>(c.coordinates()),
  			      const_cast<double*>(d.coordinates()),
  			      const_cast<double*>(b.coordinates()));
  const double abc = orient2d(const_cast<double*>(a.coordinates()),
  			      const_cast<double*>(b.coordinates()),
  			      const_cast<double*>(c.coordinates()));
  const double abd = orient2d(const_cast<double*>(a.coordinates()),
  			      const_cast<double*>(b.coordinates()),
  			      const_cast<double*>(d.coordinates()));

  Point pt;

  if (cda*cdb < 0.0 and abc*abd < 0.0) // If equality, then they align
  {
    // We have intersection (see Shewchuck Lecture Notes on Geometric
    // Robustness).

    // Robust determinant calculation (see orient2d routine). This is
    // way more involved, but skip for now.  Even Shewchuk (Lecture
    // Notes on Geometric Robustness, Apr 15, 2013) says this is a
    // difficult computation and may need exact arithmetic.
    const double detleft = (b[0]-a[0]) * (d[1]-c[1]);
    const double detright = (b[1]-a[1]) * (d[0]-c[0]);
    const double det = detleft - detright;

    // If the determinant is zero, then ab || cd. However, the
    // function should be used together with a predicate, which must
    // be robust. Hence, this routine must return a point.
    // if (std::abs(det) < 1.1*DOLFIN_EPS) // if exactly zero then ab || cd
    // {
    // }

    const double alpha = cda / det;

    pt = a + alpha*(b - a);

    // If alpha is close to 1, then pt is close to b. Repeat the
    // calculation with the points swapped. This is probably not the
    // way to do it.
    if (std::abs(1-alpha) < DOLFIN_EPS)
      pt = b + (1-alpha)*(a - b);

    if (std::abs(det) < DOLFIN_EPS)
    {
      std::cout.precision(15);
      std::cout << a[0]<<' '<<a[1]<<'\n'
    		<<b[0]<<' '<<b[1]<<'\n'
    		<<c[0]<<' '<<c[1]<<'\n'
    		<<d[0]<<' '<<d[1]<<'\n';
      std::cout << cda <<' '<<cdb<<' '<<abc << ' ' << abd<<'\n';
      std::cout << "det " << det << " ("<<detleft<<" "<<detright<<'\n'
    		<< "alpha " << alpha << '\n'
    		<< "point    plot("<<pt[0]<<','<<pt[1]<<",'*');\n";
      Point alt = a + (1-alpha)*(b - a);
      std::cout << "alt point    plot("<<alt[0]<<','<<alt[1]<<",'o');\n";
    }
  }
  else if (cda == 0)
  {
    return a; // a is on top of cd
  }
  else if (cdb == 0)
  {
    return b; // b is on top of cd
  }
  else if (abc == 0)
  {
    return c; // c is on top of ab
  }
  else if (abd == 0)
  {
    return d; // d is on top of ab
  }
  else {
    // No intersection, but return a point anyway? If we end up here
    // we may have a conflict of interest between the predicate and
    // this function
    // pt = 0.25*(a + b + c + d);
    dolfin_error("IntersectionTriangulation.cpp",
		 "intersection_edge_edge_2d function",
		 "No intersection found");
  }

  return CHECK_CGAL(pt, cgal_intersection_edge_edge_2d(a, b, c, d));
}
//-----------------------------------------------------------------------------
Point IntersectionTriangulation::_intersection_edge_edge(const Point& a,
                                                         const Point& b,
                                                         const Point& c,
                                                         const Point& d)
{
  dolfin_error("IntersectionTriangulation.cpp",
	       "intersection_edge_edge function",
	       "Not yet implemented in 3D");

  Point p;
  return p;
}
//-----------------------------------------------------------------------------
Point IntersectionTriangulation::_intersection_face_edge(const Point& r,
                                                         const Point& s,
                                                         const Point& t,
                                                         const Point& a,
                                                         const Point& b)
{
  // This standard edge face intersection test is as follows:
  // - Check if end points of the edge (a,b) on opposite side of plane
  // given by the face (r,s,t)
  // - If we have sign change, compute intersection with plane.
  // - Check if computed point is on triangle given by face.

  // If the edge and the face are in the same plane, we return false
  // and leave this to the edge-edge intersection test.

  // Tolerance for edge and face in plane (topologically 2D problem)
  //const double top_2d_tol = DOLFIN_EPS_LARGE;

  // Compute normal
  const Point rs = s - r;
  const Point rt = t - r;
  Point n = rs.cross(rt);
  n /= n.norm();

  // Check sign change (note that if either dot product is zero it's
  // orthogonal)
  const double da = n.dot(a - r);
  const double db = n.dot(b - r);

  // // Note: if da and db we may have edge intersection (detected in
  // // other routine)
  // if (da*db > 0)
  //   return false;

  // Face and edge are in topological 2d: taken care of in edge-edge
  // intersection or point in simplex.
  const double sum = std::abs(da) + std::abs(db);
  // if (sum < top_2d_tol)
  //   return false;

  // Calculate intersection
  const Point pt = a + std::abs(da) / sum * (b - a);

  return CHECK_CGAL(pt, cgal_intersection_face_edge_2d(r, s, t, a, b));
}
//------------------------------------------------------------------------------
