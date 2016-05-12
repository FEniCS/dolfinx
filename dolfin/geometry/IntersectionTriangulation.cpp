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
// Last changed: 2016-05-08

#include <dolfin/mesh/MeshEntity.h>
#include "predicates.h"
#include "CGALExactArithmetic.h"
#include "GeometryDebugging.h"
#include "CollisionDetection.h"
#include "IntersectionTriangulation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// High-level intersection triangulation functions
//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
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
std::vector<std::vector<Point>>
IntersectionTriangulation::triangulate(const std::vector<Point>& points_0,
                                       const std::vector<Point>& points_1,
                                       std::size_t gdim)
{
  // Get topological dimensions
  const std::size_t d0 = points_0.size() - 1;
  const std::size_t d1 = points_1.size() - 1;

  // Pick correct specialized implementation
  if (d0 == 2 && d1 == 1)
    return std::vector<std::vector<Point>>(1, triangulate_triangle_segment(points_0[0],
									   points_0[1],
									   points_0[2],
									   points_1[0],
									   points_1[1],
									   gdim));

  if (d0 == 2 && d1 == 2)
    return triangulate_triangle_triangle(points_0[0],
                                         points_0[1],
                                         points_0[2],
                                         points_1[0],
                                         points_1[1],
                                         points_1[2],
					 gdim);

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
               "Not implemented for dimensions %d / %d and geometric dimension %d", d0, d1, gdim);

  return std::vector<std::vector<Point>>();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
IntersectionTriangulation::triangulate
(const MeshEntity &entity,
 const std::vector<std::vector<Point>> &triangulation,
 std::size_t tdim)
{
  // Compute the triangulation of the intersection of the cell and the
  // simplices of the flat triangulation vector with topology tdim.

  std::vector<std::vector<Point>> total_triangulation;

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

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size(); ++i)
  {
    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < tdim + 1; ++j)
      simplex[j] = triangulation[i][j];

    // Compute intersection
    const std::vector<std::vector<Point>> local_triangulation
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
IntersectionTriangulation::triangulate
(const MeshEntity &entity,
 const std::vector<std::vector<Point>>& triangulation,
 const std::vector<Point>& normals,
 std::vector<std::vector<Point>>& intersection_triangulation,
 std::vector<Point>& intersection_normals,
 std::size_t tdim)
{
  // Compute the triangulation of the intersection of the cell and the
  // simplices of the flat triangulation vector with topology tdim.

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

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size(); ++i)
  {
    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < tdim + 1; ++j)
      simplex[j] = triangulation[i][j];

    // Compute intersection
    const std::vector<std::vector<Point>> local_triangulation
      = triangulate(simplex_entity, simplex, gdim);

    // Add these to the net triangulation
    intersection_triangulation.insert(intersection_triangulation.end(),
                                      local_triangulation.begin(),
                                      local_triangulation.end());

    // Add the normal
    intersection_normals.resize(intersection_normals.size() + local_triangulation.size(),
                                normals[i]);
  }
}
//-----------------------------------------------------------------------------
// Low-level intersection triangulation functions
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::triangulate_segment_segment(const Point& p0,
						       const Point& p1,
						       const Point& q0,
						       const Point& q1,
						       std::size_t gdim)
{
  switch (gdim)
  {
  case 1:
    return triangulate_segment_segment_1d(p0[0], p1[0], q0[0], q1[0]);
  case 2:
    return triangulate_segment_segment_2d(p0, p1, q0, q1);
  case 3:
    return triangulate_segment_segment_3d(p0, p1, q0, q1);
  default:
    dolfin_error("IntersectionTriangulation.cpp",
		 "compute segment-segment collision",
		 "Unknown dimension %d.", gdim);
  }
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::triangulate_triangle_segment(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& q0,
							const Point& q1,
							std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return triangulate_triangle_segment_2d(p0, p1, p2, q0, q1);
  case 3:
    return triangulate_triangle_segment_3d(p0, p1, p2, q0, q1);
  default:
    dolfin_error("IntersectionTriangulation.cpp",
		 "compute segment-segment collision",
		 "Unknown dimension %d.", gdim);
  }
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
IntersectionTriangulation::triangulate_triangle_triangle(const Point& p0,
							 const Point& p1,
							 const Point& p2,
							 const Point& q0,
							 const Point& q1,
							 const Point& q2,
							 std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return triangulate_triangle_triangle_2d(p0, p1, p2, q0, q1, q2);
  case 3:
    return triangulate_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);
  default:
    dolfin_error("IntersectionTriangulation.cpp",
		 "compute segment-segment collision",
		 "Unknown dimension %d.", gdim);
  }
  return std::vector<std::vector<Point>>();
}
//-----------------------------------------------------------------------------
// Implementation of triangulation functions
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::_triangulate_segment_segment_1d(double p0,
							   double p1,
							   double q0,
							   double q1)
{
  dolfin_error("IntersectionTriangulation.cpp",
	       "compute segment-segment collision",
	       "Not implemented for dimension 1.");

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::_triangulate_segment_segment_2d(const Point& p0,
							   const Point& p1,
							   const Point& q0,
							   const Point& q1)
{
  // Shewchuk style
  const double cda = orient2d(const_cast<double*>(q0.coordinates()),
			      const_cast<double*>(q1.coordinates()),
			      const_cast<double*>(p0.coordinates()));
  const double cdb = orient2d(const_cast<double*>(q0.coordinates()),
			      const_cast<double*>(q1.coordinates()),
			      const_cast<double*>(p1.coordinates()));
  const double abc = orient2d(const_cast<double*>(p0.coordinates()),
			      const_cast<double*>(p1.coordinates()),
			      const_cast<double*>(q0.coordinates()));
  const double abd = orient2d(const_cast<double*>(p0.coordinates()),
			      const_cast<double*>(p1.coordinates()),
			      const_cast<double*>(q1.coordinates()));
  if (cda == 0)
    return std::vector<Point>(1, p0); // p0 is on top of cd
  else if (cdb == 0)
    return std::vector<Point>(1, p1); // p1 is on top of cd
  else if (abc == 0)
    return std::vector<Point>(1, q0); // q0 is on top of ab
  else if (abd == 0)
    return std::vector<Point>(1, q1); // q1 is on top of ab
  else {
    // We assume we have an intersection. We would like to have a
    // robust determinant calculation (see orient2d routine). Since
    // this is way more involved we skip this for now. Note that
    // even Shewchuk (Lecture Notes on Geometric Robustness, Apr 15,
    // 2013) says the determinant calculation is a difficult and may
    // need exact arithmetic.
    const double detleft = (p1[0]-p0[0]) * (q1[1]-q0[1]);
    const double detright = (p1[1]-p0[1]) * (q1[0]-q0[0]);
    const double det = detleft - detright;

    // If the determinant is zero, then ab || cd
    if (std::abs(det) < DOLFIN_EPS)
    {
      // FIXME: implement this
      dolfin_error("IntersectionTriangulation.cpp",
		   "compute segment-segment triangulation ",
		   "Intersection when segments are parallel not implemented.");
    }

    const double alpha = cda / det;
    Point point = p0 + alpha*(p1 - p0);

    // If alpha is close to 1, then pt is close to b. Repeat the
    // calculation with the points swapped. This is probably not the
    // way to do it.
    if (std::abs(1-alpha) < DOLFIN_EPS)
      point = p1 + (1-alpha)*(p0 - p1);
    return std::vector<Point>(1, point);
  }
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::_triangulate_segment_segment_3d(const Point& p0,
							   const Point& p1,
							   const Point& q0,
							   const Point& q1)
{
  dolfin_error("IntersectionTriangulation.cpp",
	       "compute segment-segment 3d collision",
	       "Not implemented.");
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::_triangulate_triangle_segment_2d(const Point& p0,
							    const Point& p1,
							    const Point& p2,
							    const Point& q0,
							    const Point& q1)
{
  std::vector<Point> points;

  // Detect edge intersection points
  if (CollisionDetection::collides_segment_segment_2d(p0, p1, q0, q1))
  {
    const std::vector<Point> ii = triangulate_segment_segment_2d(p0, p1, q0, q1);
    dolfin_assert(ii.size());
    points.insert(points.end(), ii.begin(), ii.end());
  }
  if (CollisionDetection::collides_segment_segment_2d(p0, p2, q0, q1))
  {
    const std::vector<Point> ii = triangulate_segment_segment_2d(p0, p2, q0, q1);
    dolfin_assert(ii.size());
    points.insert(points.end(), ii.begin(), ii.end());
  }
  if (CollisionDetection::collides_segment_segment_2d(p1, p2, q0, q1))
  {
    const std::vector<Point> ii = triangulate_segment_segment_2d(p1, p2, q0, q1);
    dolfin_assert(ii.size());
    points.insert(points.end(), ii.begin(), ii.end());
  }

  if (points.size() == 0)
  {
    // If we get zero intersection points, then both segment ends must
    // be inside. Note that we here assume that we have called this
    // routine _after_ a suitable predicate from CollisionDetection,
    // meaning we know that the triangle and segment collides
    return std::vector<Point>{{q0, q1}};
  }
  else if (points.size() == 1)
  {
    // If we get one intersection point, find the segment end point
    // which is inside the triangle.
    if (CollisionDetection::collides_triangle_point_2d(p0, p1, p2, q0))
      return std::vector<Point>{{ q0, points[0] }};
    else
      return std::vector<Point>{{ q1, points[0] }};
  }
  else if (points.size() == 2)
  {
    // If we get two intersection points, this is the intersection
    return points;
  }
  else
  {
    dolfin_error("IntersectionTriangulation.cpp",
		 "compute triangle-segment 2d triangulation ",
		 "Unknown number of points %d", points.size());
  }
  return points;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionTriangulation::_triangulate_triangle_segment_3d(const Point& p0,
							    const Point& p1,
							    const Point& p2,
							    const Point& q0,
							    const Point& q1)
{
  dolfin_error("IntersectionTriangulation.cpp",
	       "compute triangle-segment 3d collision",
	       "Not implemented.");
  return std::vector<Point>();
}

//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
IntersectionTriangulation::_triangulate_triangle_triangle_2d(const Point& p0,
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

  if (points.size() == 3)
  {
    // Triangle is inside
    return std::vector<std::vector<Point>>(1, points);
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
      if (CollisionDetection::collides_segment_segment_2d(p0, q0, p1, q1))
      {
	const std::vector<Point> ii = triangulate_segment_segment_2d(p0, q0, p1, q1);
	dolfin_assert(ii.size());
	points.insert(points.end(), ii.begin(), ii.end());
      }
    }
  }

  // Remove duplicate points
  std::vector<Point> tmp;
  tmp.reserve(points.size());

  for (std::size_t i = 0; i < points.size(); ++i)
  {
    bool different = true;
    for (std::size_t j = i+1; j < points.size(); ++j)
      if ((points[i] - points[j]).norm() < DOLFIN_EPS)
      {
  	different = false;
  	break;
      }
    if (different)
      tmp.push_back(points[i]);
  }
  points = tmp;

  // If the number of points is less than four, then these form the
  // triangulation
  if (points.size() < 4)
    return std::vector<std::vector<Point>>(1, points);

  // If 4 or greater, do graham scan
  dolfin_assert(points.size() == 4 or
		points.size() == 6);
  const std::vector<std::vector<Point>> triangulation = graham_scan(points);

  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
IntersectionTriangulation::_triangulate_triangle_triangle_3d(const Point& p0,
							     const Point& p1,
							     const Point& p2,
							     const Point& q0,
							     const Point& q1,
							     const Point& q2)
{
  dolfin_error("IntersectionTriangulation.cpp",
	       "compute triangle-triangle 3d collision",
	       "Not implemented.");
  return std::vector<std::vector<Point>>();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
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
  const double same_point_tol = DOLFIN_EPS;

  // Tolerance for small triangle (could be improved by identifying
  // sliver and small triangles)
  const double tri_det_tol = DOLFIN_EPS;

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
    if (CollisionDetection::collides_triangle_segment_3d(tri[0], tri[1], tri[2],
							 tet[tet_edges[e][0]],
							 tet[tet_edges[e][1]]))
    {
      const std::vector<Point> ii = triangulate_triangle_segment_3d(tri[0], tri[1], tri[2],
								    tet[tet_edges[e][0]],
								    tet[tet_edges[e][1]]);
      dolfin_assert(ii.size());
      points.insert(points.end(), ii.begin(), ii.end());
    }

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

  const std::array<std::array<std::size_t, 2>, 3> tri_edges = {{ {0, 1},
								 {0, 2},
								 {1, 2} }};

  for (std::size_t f = 0; f < 4; ++f)
    for (std::size_t e = 0; e < 3; ++e)
      if (CollisionDetection::collides_triangle_segment_3d(tet[tet_faces[f][0]],
							   tet[tet_faces[f][1]],
							   tet[tet_faces[f][2]],
							   tri[tri_edges[e][0]],
							   tri[tri_edges[e][1]]))
      {
	const std::vector<Point> ii = triangulate_triangle_segment_3d(tet[tet_faces[f][0]],
								      tet[tet_faces[f][1]],
								      tet[tet_faces[f][2]],
								      tri[tri_edges[e][0]],
								      tri[tri_edges[e][1]]);
	dolfin_assert(ii.size());
	points.insert(points.end(), ii.begin(), ii.end());
      }

  // // FIXME: segment-segment intersection should not be needed if
  // // triangle-segment intersection doesn't miss this
  // for (std::size_t f = 0; f < 6; ++f)
  // {
  //   if (CollisionDetection::collides_segment_segment_3d(tet[tet_edges[f][0]],
  // 							tet[tet_edges[f][1]],
  // 							tri[0], tri[1]))
  //     points.push_back(triangulate_segment_segment_3d(tet[tet_edges[f][0]],
  // 						      tet[tet_edges[f][1]],
  // 						      tri[0], tri[1]));

  //   if (CollisionDetection::collides_segment_segment_3d(tet[tet_edges[f][0]],
  // 							tet[tet_edges[f][1]],
  // 							tri[0], tri[2]))
  //     points.push_back(_intersection_segment_segment_3d(tet[tet_edges[f][0]],
  // 							tet[tet_edges[f][1]],
  // 							tri[0], tri[2]));

  //   if (CollisionDetection::collides_segment_segment_3d(tet[tet_edges[f][0]],
  // 							tet[tet_edges[f][1]],
  // 							tri[1], tri[2]))
  //     points.push_back(_intersection_segment_segment_3d(tet[tet_edges[f][0]],
  // 							tet[tet_edges[f][1]],
  // 							tri[1], tri[2]));
  // }


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
    return std::vector<std::vector<Point>>();

  std::vector<std::vector<Point>> triangulation;

  Point n = (points[2] - points[0]).cross(points[1] - points[0]);
  const double det = n.norm();
  n /= det;

  if (points.size() == 3) {
    // Include if determinant is sufficiently large
    if (det > tri_det_tol)
    {
      // One triangle
      triangulation.assign(1, {{ points[0], points[1], points[2] }});
    }
    return triangulation;
  }

  // Tessellate as in the triangle-triangle intersection case: First
  // sort points using a Graham scan, then connect to form triangles.
  return graham_scan(points);
}
//-----------------------------------------------------------------------------
std::vector<std::vector<Point>>
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
  //const double tet_det_tol = DOLFIN_EPS_LARGE;

  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  const double same_point_tol = DOLFIN_EPS_LARGE;

  // Tolerance for small triangle (could be improved by identifying
  // sliver and small triangles)
  //const double tri_det_tol = DOLFIN_EPS_LARGE;

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
      if (CollisionDetection::collides_triangle_segment_3d(tet_0[faces_0[f][0]],
							   tet_0[faces_0[f][1]],
							   tet_0[faces_0[f][2]],
							   tet_1[edges_1[e][0]],
							   tet_1[edges_1[e][1]]))
      {
	const std::vector<Point> ii = triangulate_triangle_segment_3d(tet_0[faces_0[f][0]],
								      tet_0[faces_0[f][1]],
								      tet_0[faces_0[f][2]],
								      tet_1[edges_1[e][0]],
								      tet_1[edges_1[e][1]]);
	points.insert(points.end(), ii.begin(), ii.end());
      }

      if (CollisionDetection::collides_triangle_segment_3d(tet_1[faces_1[f][0]],
							   tet_1[faces_1[f][1]],
							   tet_1[faces_1[f][2]],
							   tet_0[edges_0[e][0]],
							   tet_0[edges_0[e][1]]))
      {
	const std::vector<Point> ii = triangulate_triangle_segment_3d(tet_1[faces_1[f][0]],
								      tet_1[faces_1[f][1]],
								      tet_1[faces_1[f][2]],
								      tet_0[edges_0[e][0]],
								      tet_0[edges_0[e][1]]);
	points.insert(points.end(), ii.begin(), ii.end());
      }
    }
  }

  // Edge edge intersection
  Point pt;
  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      if (CollisionDetection::collides_segment_segment_3d(tet_0[edges_0[i][0]],
							  tet_0[edges_0[i][1]],
							  tet_1[edges_1[j][0]],
							  tet_1[edges_1[j][1]]))
      {
	const std::vector<Point> ii = triangulate_segment_segment_3d(tet_0[edges_0[i][0]],
								     tet_0[edges_0[i][1]],
								     tet_1[edges_1[j][0]],
								     tet_1[edges_1[j][1]]);
	points.insert(points.end(), ii.begin(), ii.end());
      }
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
    return std::vector<std::vector<Point>>();

  // Start forming a tessellation
  if (points.size() == 4)
  {
    // FIXME: We could check that the volume is sufficiently large
    return std::vector<std::vector<Point>>(1, points);
  }

  // Points forming the tetrahedral partitioning of the polyhedron
  std::vector<std::vector<Point>> triangulation;

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
  	  Point n = (points[j] - points[i]).cross(points[k] - points[i]);
  	  const double tridet = n.norm();

	  // FIXME: Here we could check that the triangle is sufficiently large
  	  // if (tridet < tri_det_tol)
          //   break;

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

  	      // FIXME: Here we could include if determinant is sufficiently large
	      triangulation.push_back(cand);
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
  	      for (std::size_t m = 0; m < coplanar.size() - 2; ++m)
  	      {
  		// Candidate tetrahedron:
  		std::vector<Point> cand(4);
  		cand[0] = points[coplanar[0]];
  		cand[1] = points[coplanar[order[m].second]];
  		cand[2] = points[coplanar[order[m + 1].second]];
  		cand[3] = polyhedroncenter;

		// FIXME: Possibly only include if tet is large enough
		triangulation.push_back(cand);
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
std::vector<std::vector<Point>>
IntersectionTriangulation::graham_scan(const std::vector<Point>& points)
{
  // NB: The input points should be unique.

  // Sometimes we can get an extra point on an edge: a-----c--b. This
  // point c may cause problems for the graham scan. To avoid this,
  // use an extra center point.  Use this center point and point no 0
  // as reference for the angle calculation
  Point pointscenter = points[0];
  for (std::size_t m = 1; m < points.size(); ++m)
    pointscenter += points[m];
  pointscenter /= points.size();

  std::vector<std::pair<double, std::size_t>> order;
  Point ref = points[0] - pointscenter;
  ref /= ref.norm();

  // Compute normal
  Point normal = (points[2] - points[0]).cross(points[1] - points[0]);
  const double det = normal.norm();
  normal /= det;

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
      if (v.dot(normal.cross(ref)) < 0)
        alpha = 2*DOLFIN_PI-alpha;
    }
    order.push_back(std::make_pair(alpha, m));
  }

  // Sort angles
  std::sort(order.begin(), order.end());

  // Tessellate
  std::vector<std::vector<Point>> triangulation(order.size() - 1);
  for (std::size_t m = 0; m < order.size()-1; ++m)
  {
    // FIXME: We could consider only triangles with area > tolerance here.
    triangulation[m] = {{ points[0],
			  points[order[m].second],
			  points[order[m + 1].second] }};
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
