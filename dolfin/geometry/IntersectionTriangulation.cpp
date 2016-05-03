// Copyright (C) 2014 Anders Logg and August Johansson
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
// Last changed: 2016-05-03

#include <dolfin/mesh/MeshEntity.h>
#include "IntersectionTriangulation.h"
#include "CGALExactArithmetic.h"
#include "CollisionDetection.h"
#include "predicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection(const MeshEntity& entity_0,
						    const MeshEntity& entity_1)
{
  switch (entity_0.dim())
  {
  case 0:
    // PointCell
    dolfin_not_implemented();
    break;
  case 1:
    // IntervalCell
    dolfin_not_implemented();
    break;
  case 2:
    // TriangleCell
    switch (entity_1.dim())
    {
    case 0:
      dolfin_not_implemented();
      break;
    case 1:
      return triangulate_intersection_triangle_interval(entity_0, entity_1);
    case 2:
      return triangulate_intersection_triangle_triangle(entity_0, entity_1);
    case 3:
      return triangulate_intersection_tetrahedron_triangle(entity_1, entity_0);
    default:
      dolfin_error("IntersectionTriangulation.cpp",
		   "triangulate intersection of entity_0 and entity_1",
		   "unknown dimension of entity_1 in TriangleCell");
    }
  case 3:
    // TetrahedronCell
    switch (entity_1.dim())
    {
    case 0:
      dolfin_not_implemented();
      break;
    case 1:
      dolfin_not_implemented();
      break;
    case 2:
      return triangulate_intersection_tetrahedron_triangle(entity_0, entity_1);
    case 3:
      return triangulate_intersection_tetrahedron_tetrahedron(entity_0, entity_1);
    default:
      dolfin_error("IntersectionTriangulation.cpp",
		   "triangulate intersection of entity_0 and entity_1",
		   "unknown dimension of entity_1 in TetrahedronCell");
    }
  default:
    dolfin_error("IntersectionTriangulation.cpp",
		 "triangulate intersection of entity_0 and entity_1",
		 "unknown dimension of entity_0");
  }
  return std::vector<double>();
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_interval_interval
(const MeshEntity& interval_0, const MeshEntity& interval_1)
{
  dolfin_assert(interval_0.mesh().topology().dim() == 1);
  dolfin_assert(interval_1.mesh().topology().dim() == 1);

  const std::size_t gdim = interval_0.mesh().geometry().dim();
  dolfin_assert(interval_1.mesh().topology().dim() == gdim);

  // Get geometry and vertex data
  std::vector<Point> inter_0(2), inter_1(2);
  for (std::size_t i = 0; i < 2; ++i)
  {
    inter_0[i] = interval_0.mesh().geometry().point(interval_0.entities(0)[i]);
    inter_1[i] = interval_1.mesh().geometry().point(interval_1.entities(0)[i]);
  }

  return triangulate_intersection_interval_interval(inter_0, inter_1, gdim);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_triangle_interval
(const MeshEntity& triangle,
 const MeshEntity& interval)
{
  dolfin_assert(triangle.mesh().topology().dim() == 2);
  dolfin_assert(interval.mesh().topology().dim() == 1);

  const std::size_t gdim = triangle.mesh().geometry().dim();
  dolfin_assert(interval.mesh().geometry().dim() == gdim);

  // Get geometry and vertex data
  std::vector<Point> tri(3);
  for (std::size_t i = 0; i < 3; ++i)
    tri[i] = triangle.mesh().geometry().point(triangle.entities(0)[i]);

  std::vector<Point> inter(2);
  for (std::size_t i = 0; i < 2; ++i)
    inter[i] = interval.mesh().geometry().point(interval.entities(0)[i]);

  return triangulate_intersection_triangle_interval(tri, inter, gdim);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_triangle_triangle
(const MeshEntity& c0, const MeshEntity& c1)
{
  // Triangulate the intersection of the two triangles c0 and c1

  dolfin_assert(c0.mesh().topology().dim() == 2);
  dolfin_assert(c1.mesh().topology().dim() == 2);

  // FIXME: Only 2D for now
  dolfin_assert(c0.mesh().geometry().dim() == 2);

  // Get geometry and vertex data
  const MeshGeometry& geometry_0 = c0.mesh().geometry();
  const MeshGeometry& geometry_1 = c1.mesh().geometry();
  const unsigned int* vertices_0 = c0.entities(0);
  const unsigned int* vertices_1 = c1.entities(0);

  std::vector<Point> tri_0(3), tri_1(3);

  for (std::size_t i = 0; i < 3; ++i)
  {
    tri_0[i] = geometry_0.point(vertices_0[i]);
    tri_1[i] = geometry_1.point(vertices_1[i]);
  }

  return triangulate_intersection_triangle_triangle(tri_0, tri_1);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_triangle
(const MeshEntity& tetrahedron, const MeshEntity& triangle)
{
  // Triangulate the intersection of a tetrahedron and a triangle

  dolfin_assert(tetrahedron.mesh().topology().dim() == 3);
  dolfin_assert(triangle.mesh().topology().dim() == 2);

  // Get geometry and vertex data
  const MeshGeometry& tet_geom = tetrahedron.mesh().geometry();
  const unsigned int* tet_vert = tetrahedron.entities(0);
  std::vector<Point> tet(4);
  for (std::size_t i = 0; i < 4; ++i)
    tet[i] = tet_geom.point(tet_vert[i]);

  const MeshGeometry& tri_geom = triangle.mesh().geometry();
  const unsigned int* tri_vert = triangle.entities(0);
  std::vector<Point> tri(3);
  for (std::size_t i = 0; i < 3; ++i)
    tri[i] = tri_geom.point(tri_vert[i]);

  return triangulate_intersection_tetrahedron_triangle(tet, tri);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_tetrahedron
(const MeshEntity& tetrahedron_0,
 const MeshEntity& tetrahedron_1)
{
  // Triangulate the intersection of the two tetrahedra

  dolfin_assert(tetrahedron_0.mesh().topology().dim() == 3);
  dolfin_assert(tetrahedron_1.mesh().topology().dim() == 3);

  // Get the vertices as points
  const MeshGeometry& geometry_0 = tetrahedron_0.mesh().geometry();
  const unsigned int* vertices_0 = tetrahedron_0.entities(0);
  const MeshGeometry& geometry_1 = tetrahedron_1.mesh().geometry();
  const unsigned int* vertices_1 = tetrahedron_1.entities(0);

  std::vector<Point> tet_0(4), tet_1(4);

  for (std::size_t i = 0; i < 4; ++i)
  {
    tet_0[i] = geometry_0.point(vertices_0[i]);
    tet_1[i] = geometry_1.point(vertices_1[i]);
  }

  return triangulate_intersection_tetrahedron_tetrahedron(tet_0, tet_1);
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection(const std::vector<Point>& s0,
                                                    std::size_t tdim0,
                                                    const std::vector<Point>& s1,
                                                    std::size_t tdim1,
                                                    std::size_t gdim)
{
  // General intersection computation of two simplices with different
  // topological dimension but the same geometrical dimension

  switch (tdim0) {
    // s0 is interval
  case 1:
    switch (tdim1) {
    case 1: // s1 is interval
      return triangulate_intersection_interval_interval(s0, s1, gdim);
    case 2: // s1 is triangle
      return triangulate_intersection_triangle_interval(s1, s0, gdim);
    case 3: // s1 is tetrahedron
      dolfin_not_implemented();
      break;
    default:
      dolfin_error("IntersectionTriangulation.cpp",
                   "triangulate intersection of two simplices s0 and s1",
                   "unknown topology %d", tdim1);
    }
    break;
    // s0 is a triangle
  case 2:
    switch (tdim1) {
    case 1: // s1 is interval
      return triangulate_intersection_triangle_interval(s0, s1, gdim);
    case 2: // s1 is triangle
      return triangulate_intersection_triangle_triangle(s0, s1);
    case 3: // s1 is tetrahedron
      return triangulate_intersection_tetrahedron_triangle(s1, s0);
    default:
      dolfin_error("IntersectionTriangulation.cpp",
                   "triangulate intersection of two simplices s0 and s1",
                   "unknown topology %d", tdim1);
    }
    break;
    // s0 is a tetrahedron
  case 3:
    switch (tdim1) {
    case 1: // s1 is interval
      dolfin_not_implemented();
      break;
    case 2: // s1 is triangle
      return triangulate_intersection_tetrahedron_triangle(s0, s1);
    case 3: // s1 is tetrahedron
      return triangulate_intersection_tetrahedron_tetrahedron(s0, s1);
    default:
      dolfin_error("IntersectionTriangulation.cpp",
                   "triangulate intersection of two simplices s0 and s1",
                   "unknown topology %d", tdim1);
    }
    break;
  default:
    dolfin_error("IntersectionTriangulation.cpp",
                 "triangulate intersection of two simplices s0 and s1",
                 "unknown topology %d", tdim0);
  }

  // We never end up here
  return std::vector<double>();
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_interval_interval
(const std::vector<Point>& interval_0,
 const std::vector<Point>& interval_1,
 std::size_t gdim)
{
  // Flat array for triangulation
  std::vector<double> triangulation;

  if (CollisionDetection::collides_edge_edge(interval_0[0], interval_0[1],
                                             interval_1[0], interval_1[1]))
  {
    // List of colliding points
    std::vector<Point> points;

    for (std::size_t i = 0; i < 2; ++i)
    {
      if (CollisionDetection::collides_interval_point(interval_0[0], interval_0[1],
                                                      interval_1[i]))
        points.push_back(interval_1[i]);
      if (CollisionDetection::collides_interval_point(interval_1[0], interval_1[1],
                                                      interval_0[i]))
        points.push_back(interval_0[i]);
    }

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

  return CHECK_CGAL(triangulation,
		    cgal_triangulate_intersection_interval_interval(interval_0, interval_1, gdim));
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_triangle_interval
(const std::vector<Point>& triangle,
 const std::vector<Point>& interval,
 std::size_t gdim)
{
  std::vector<double> triangulation;
  std::vector<Point> points;

  // // Detect edge intersection points
  // Point pt;
  // if (intersection_edge_edge(triangle[0], triangle[1],
  //                            interval[0], interval[1],
  //                            pt))
  //   points.push_back(pt);
  // if (intersection_edge_edge(triangle[0], triangle[2],
  //                            interval[0], interval[1],
  //                            pt))
  //   points.push_back(pt);
  // if (intersection_edge_edge(triangle[1], triangle[2],
  //                            interval[0], interval[1],
  //                            pt))
  //   points.push_back(pt);

  // Detect edge intersection points
  if (CollisionDetection::collides_edge_edge(triangle[0], triangle[1],
					     interval[0], interval[1]))
    points.push_back(intersection_edge_edge(triangle[0], triangle[1],
					    interval[0], interval[1]));
  if (CollisionDetection::collides_edge_edge(triangle[0], triangle[2],
					     interval[0], interval[1]))
    points.push_back(intersection_edge_edge(triangle[0], triangle[2],
					    interval[0], interval[1]));
  if (CollisionDetection::collides_edge_edge(triangle[1], triangle[2],
					     interval[0], interval[1]))
    points.push_back(intersection_edge_edge(triangle[1], triangle[2],
					    interval[0], interval[1]));


  // If we get zero intersection points, then both interval ends must
  // be inside
  // FIXME: can we really use two different types of intersection tests: intersection_edge_edge above and Collides here?
  if (points.size() == 0)
  {
    if (CollisionDetection::collides_triangle_point(triangle[0],
                                                    triangle[1],
                                                    triangle[2],
                                                    interval[0]) and
        CollisionDetection::collides_triangle_point(triangle[0],
                                                    triangle[1],
                                                    triangle[2],
                                                    interval[1]))
    {
      triangulation.resize(2*gdim);
      for (std::size_t d = 0; d < gdim; ++d)
      {
        triangulation[d] = interval[0][d];
        triangulation[gdim+d] = interval[1][d];
      }
      return triangulation;
    }
  }

  // If we get one intersection point, find the interval end point
  // which is inside the triangle. Note that this points should
  // absolutely not be the same point as we found above. This can
  // happen since we use different types of tests here and above.
  if (points.size() == 1)
  {
    for (std::size_t k = 0; k < 2; ++k)
    {
      // Make sure the point interval[k] is not points[0]
      if ((interval[k]-points[0]).norm() > DOLFIN_EPS_LARGE and
  	  CollisionDetection::collides_triangle_point(triangle[0],
  						      triangle[1],
  						      triangle[2],
  						      interval[k]))
      {
        triangulation.resize(2*gdim);
        for (std::size_t d = 0; d < gdim; ++d)
        {
          triangulation[d] = points[0][d];
          triangulation[gdim+d] = interval[k][d];
        }
        return triangulation;
      }
    }
  }

  // If we get two intersection points, triangulate this line.
  if (points.size() == 2)
  {
    triangulation.resize(2*gdim);
    for (std::size_t d = 0; d < gdim; ++d)
    {
      triangulation[d] = points[0][d];
      triangulation[gdim+d] = points[1][d];
    }
    return triangulation;
  }

  return CHECK_CGAL(triangulation,
		    cgal_triangulate_intersection_triangle_interval(triangle, interval, gdim));
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_triangle_triangle
(const std::vector<Point>& tri_0,
 const std::vector<Point>& tri_1)
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


  // Create empty list of collision points
  std::vector<Point> points;

  // // Find all vertex-cell collisions
  // for (std::size_t i = 0; i < 3; i++)
  // {
  //   // Note: this routine is changed to being public:
  //   if (CollisionDetection::collides_triangle_point_2d(tri_1[0],
  // 						       tri_1[1],
  // 						       tri_1[2],
  // 						       tri_0[i]))
  //     points.push_back(tri_0[i]);

  //   if (CollisionDetection::collides_triangle_point_2d(tri_0[0],
  // 						       tri_0[1],
  // 						       tri_0[2],
  // 						       tri_1[i]))
  //     points.push_back(tri_1[i]);
  // }

  double t0[3][2] = {{tri_0[0][0], tri_0[0][1]},
  		     {tri_0[1][0], tri_0[1][1]},
  		     {tri_0[2][0], tri_0[2][1]}};
  double t1[3][2] = {{tri_1[0][0], tri_1[0][1]},
  		     {tri_1[1][0], tri_1[1][1]},
  		     {tri_1[2][0], tri_1[2][1]}};

  // Find all vertex-cell collisions
  const int s0 = std::signbit(orient2d(t0[0], t0[1], t0[2])) == true ? -1 : 1;
  const int s1 = std::signbit(orient2d(t1[0], t1[1], t1[2])) == true ? -1 : 1;

  // std::cout << "signs " << s0 << ' ' << s1 << std::endl;

  // Note: should have >= to allow for 2 meshes, 2 elements each, rotation 100
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

  // std::cout << "after triangle_point: ";
  // for (const auto p: points)
  //   std::cout << p[0]<<' '<<p[1]<< "     ";
  // std::cout << '\n';



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
      // Point point;
      // if (intersection_edge_edge_2d(p0, q0, p1, q1, point))
      //   points.push_back(point);
      if (CollisionDetection::collides_edge_edge(p0, q0, p1, q1))
	points.push_back(intersection_edge_edge_2d(p0, q0, p1, q1));
    }
  }

  // std::cout << "after edge-edge: ";
  // for (const auto p: points)
  //   std::cout << p[0]<<' '<<p[1]<< "     ";
  // std::cout << '\n';



  // // The function intersection_edge_edge only gives one point. Thus,
  // // check edge-point intersections separately.
  // for (std::size_t i0 = 0; i0 < 3; ++i0)
  // {
  //   const std::size_t j0 = (i0 + 1) % 3;
  //   const Point& p0 = tri_0[i0];
  //   const Point& q0 = tri_0[j0];
  //   for (std::size_t i1 = 0; i1 < 3; ++i1)
  //   {
  //     const Point& point_1 = tri_1[i1];
  //     if (CollisionDetection::collides_interval_point(p0, q0, point_1))
  // 	points.push_back(point_1);
  //   }
  // }

  // // std::cout << "after first edge-point: ";
  // // for (const auto p: points)
  // //   std::cout << p[0]<<' '<<p[1]<< "     ";
  // // std::cout << '\n';



  // for (std::size_t i0 = 0; i0 < 3; ++i0)
  // {
  //   const std::size_t j0 = (i0 + 1) % 3;
  //   const Point& p1 = tri_1[i0];
  //   const Point& q1 = tri_1[j0];
  //   for (std::size_t i1 = 0; i1 < 3; ++i1)
  //   {
  //     const Point& point_0 = tri_0[i1];
  //     if (CollisionDetection::collides_interval_point(p1, q1, point_0))
  // 	points.push_back(point_0);
  //   }
  // }


  // // std::cout << "after second edge-point: ";
  // // for (const auto p: points)
  // //   std::cout << p[0]<<' '<<p[1]<< "     ";
  // // std::cout << '\n';



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

  // If the number of points > 3, form triangles using center
  // point. This avoids skinny triangles in multimesh.

  // std::cout << "net points: ";
  // for (const auto p: points)
  // {
  //   std::cout << p[0]<<' '<<p[1]<< "     ";
  // }
  // std::cout << '\n';



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
  const Point& p0 = points[i_min];
  for (std::size_t i = 0; i < points.size() - 2; i++)
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

  // std::cout << "min angle " << minimum_angle(&triangulation[0], &triangulation[2], &triangulation[4]) << std::endl;

  return CHECK_CGAL(triangulation,
		    cgal_triangulate_intersection_triangle_triangle(tri_0, tri_1));


  // // Create triangulation using center point.
  // Point c = points[0];
  // for (std::size_t i = 1; i < points.size(); ++i)
  //   c += points[i];
  // c /= points.size();

  // // Calculate and store angles
  // std::vector<std::pair<double, std::size_t>> order(points.size());
  // for (std::size_t i = 0; i < points.size(); ++i)
  // {
  //   const Point v = points[i] - c;
  //   const double alpha = atan2(v.y(), v.x());
  //   order[i] = std::make_pair(alpha, i);
  // }

  // // Sort points based on angle
  // std::sort(order.begin(), order.end());

  // // Put first points last for cyclic use
  // order.push_back(order.front());

  // // Form the triangulation
  // //triangulation.reserve(2*3*points.size());
  // triangulation.resize(2*3*points.size());

  // for (std::size_t i = 0; i < points.size(); ++i)
  // {
  //   const Point& p1 = points[order[i].second];
  //   const Point& p2 = points[order[i + 1].second];
  //   triangulation[6*i] = c.x();
  //   triangulation[6*i+1] = c.y();
  //   triangulation[6*i+2] = p1.x();
  //   triangulation[6*i+3] = p1.y();
  //   triangulation[6*i+4] = p2.x();
  //   triangulation[6*i+5] = p2.y();

  //   //std::cout << "min angle " << minimum_angle(&triangulation[6*i+0], &triangulation[6*i+2], &triangulation[6*i+4]) << std::endl;
  // }

  // return triangulation;

}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_tetrahedron
(const std::vector<Point>& tet_0,
 const std::vector<Point>& tet_1)
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
  for (int i = 0; i<4; ++i)
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
    for (std::size_t f = 0; f < 4; ++f)
    {
      // Point pta;
      // if (intersection_face_edge(tet_0[faces_0[f][0]],
      // 				 tet_0[faces_0[f][1]],
      // 				 tet_0[faces_0[f][2]],
      // 				 tet_1[edges_1[e][0]],
      // 				 tet_1[edges_1[e][1]],
      // 				 pta))
      // 	points.push_back(pta);

      // Point ptb;
      // if (intersection_face_edge(tet_1[faces_1[f][0]],
      // 				 tet_1[faces_1[f][1]],
      // 				 tet_1[faces_1[f][2]],
      // 				 tet_0[edges_0[e][0]],
      // 				 tet_0[edges_0[e][1]],
      // 				 ptb))
      // 	points.push_back(ptb);

      if (CollisionDetection::collides_triangle_interval(tet_0[faces_0[f][0]],
							 tet_0[faces_0[f][1]],
							 tet_0[faces_0[f][2]],
							 tet_1[edges_1[e][0]],
							 tet_1[edges_1[e][1]]))
	points.push_back(intersection_face_edge(tet_0[faces_0[f][0]],
						tet_0[faces_0[f][1]],
						tet_0[faces_0[f][2]],
						tet_1[edges_1[e][0]],
						tet_1[edges_1[e][1]]));

      if (CollisionDetection::collides_triangle_interval(tet_1[faces_1[f][0]],
							 tet_1[faces_1[f][1]],
							 tet_1[faces_1[f][2]],
							 tet_0[edges_0[e][0]],
							 tet_0[edges_0[e][1]]))
	points.push_back(intersection_face_edge(tet_1[faces_1[f][0]],
						tet_1[faces_1[f][1]],
						tet_1[faces_1[f][2]],
						tet_0[edges_0[e][0]],
						tet_0[edges_0[e][1]]));
    }

  // Edge edge intersection
  Point pt;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      // if (intersection_edge_edge(tet_0[edges_0[i][0]],
      // 				 tet_0[edges_0[i][1]],
      // 				 tet_1[edges_1[j][0]],
      // 				 tet_1[edges_1[j][1]],
      // 				 pt))
      // 	points.push_back(pt);
      if (CollisionDetection::collides_edge_edge(tet_0[edges_0[i][0]],
						 tet_0[edges_0[i][1]],
						 tet_1[edges_1[j][0]],
						 tet_1[edges_1[j][1]]))
	points.push_back(intersection_edge_edge(tet_0[edges_0[i][0]],
						tet_0[edges_0[i][1]],
						tet_1[edges_1[j][0]],
						tet_1[edges_1[j][1]]));

  // Remove duplicate nodes
  std::vector<Point> tmp;
  tmp.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    bool different=true;
    for (std::size_t j = i+1; j < points.size(); ++j)
    {
      if ((points[i] - points[j]).norm() < same_point_tol) {
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
    for (std::size_t j = i+1; j < N; ++j)
      for (std::size_t k = 0; k < N; ++k)
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


  return triangulation;


}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_triangle
(const std::vector<Point>& tet,
 const std::vector<Point>& tri)
{
  // This code mimics the
  // triangulate_intersection_tetrahedron_tetrahedron and the
  // triangulate_intersection_tetrahedron_tetrahedron_triangle_codes:
  // we first identify triangle nodes in the tetrahedra. Them we
  // continue with edge-face detection for the four faces of the
  // tetrahedron and the triangle. The points found are used to form a
  // triangulation by first sorting them using a Graham scan.

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

  // Point pt;
  // for (std::size_t e = 0; e < 6; ++e)
  //   if (intersection_face_edge(tri[0], tri[1], tri[2],
  // 			       tet[tet_edges[e][0]],
  // 			       tet[tet_edges[e][1]],
  // 			       pt))
  //     points.push_back(pt);

  for (std::size_t e = 0; e < 6; ++e)
    if (CollisionDetection::collides_triangle_interval(tri[0], tri[1], tri[2],
						       tet[tet_edges[e][0]],
						       tet[tet_edges[e][1]]))
      points.push_back(intersection_face_edge(tri[0], tri[1], tri[2],
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

  // for (std::size_t f = 0; f < 4; ++f)
  // {
  //   if (intersection_face_edge(tet[tet_faces[f][0]],
  // 			       tet[tet_faces[f][1]],
  // 			       tet[tet_faces[f][2]],
  // 			       tri[0], tri[1],
  //                              pt))
  //     points.push_back(pt);
  //   if (intersection_face_edge(tet[tet_faces[f][0]],
  // 			       tet[tet_faces[f][1]],
  // 			       tet[tet_faces[f][2]],
  // 			       tri[0], tri[2],
  //                              pt))
  //     points.push_back(pt);
  //   if (intersection_face_edge(tet[tet_faces[f][0]],
  // 			       tet[tet_faces[f][1]],
  // 			       tet[tet_faces[f][2]],
  // 			       tri[1], tri[2],
  //                              pt))
  //     points.push_back(pt);
  // }

  for (std::size_t f = 0; f < 4; ++f)
  {
    if (CollisionDetection::collides_triangle_interval(tet[tet_faces[f][0]],
						       tet[tet_faces[f][1]],
						       tet[tet_faces[f][2]],
						       tri[0], tri[1]))
      points.push_back(intersection_face_edge(tet[tet_faces[f][0]],
					      tet[tet_faces[f][1]],
					      tet[tet_faces[f][2]],
					      tri[0], tri[1]));

    if (CollisionDetection::collides_triangle_interval(tet[tet_faces[f][0]],
						       tet[tet_faces[f][1]],
						       tet[tet_faces[f][2]],
						       tri[0], tri[2]))
      points.push_back(intersection_face_edge(tet[tet_faces[f][0]],
					      tet[tet_faces[f][1]],
					      tet[tet_faces[f][2]],
					      tri[0], tri[2]));

    if (CollisionDetection::collides_triangle_interval(tet[tet_faces[f][0]],
						       tet[tet_faces[f][1]],
						       tet[tet_faces[f][2]],
						       tri[1], tri[2]))
      points.push_back(intersection_face_edge(tet[tet_faces[f][0]],
					      tet[tet_faces[f][1]],
					      tet[tet_faces[f][2]],
					      tri[1], tri[2]));
  }

  // // edge edge intersection
  // for (std::size_t f = 0; f < 6; ++f)
  // {
  //   if (intersection_edge_edge(tet[tet_edges[f][0]],
  // 			       tet[tet_edges[f][1]],
  // 			       tri[0], tri[1],
  // 			       pt))
  //     points.push_back(pt);
  //   if (intersection_edge_edge(tet[tet_edges[f][0]],
  // 			       tet[tet_edges[f][1]],
  // 			       tri[0], tri[2],
  // 			       pt))
  //     points.push_back(pt);
  //   if (intersection_edge_edge(tet[tet_edges[f][0]],
  // 			       tet[tet_edges[f][1]],
  // 			       tri[1], tri[2],
  // 			       pt))
  //     points.push_back(pt);
  // }

  // edge edge intersection
  for (std::size_t f = 0; f < 6; ++f)
  {
    if (CollisionDetection::collides_edge_edge(tet[tet_edges[f][0]],
					       tet[tet_edges[f][1]],
					       tri[0], tri[1]))
      points.push_back(intersection_edge_edge(tet[tet_edges[f][0]],
					      tet[tet_edges[f][1]],
					      tri[0], tri[1]));

    if (CollisionDetection::collides_edge_edge(tet[tet_edges[f][0]],
					       tet[tet_edges[f][1]],
					       tri[0], tri[2]))
      points.push_back(intersection_edge_edge(tet[tet_edges[f][0]],
					      tet[tet_edges[f][1]],
					      tri[0], tri[2]));

    if (CollisionDetection::collides_edge_edge(tet[tet_edges[f][0]],
					       tet[tet_edges[f][1]],
					       tri[1], tri[2]))
      points.push_back(intersection_edge_edge(tet[tet_edges[f][0]],
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
IntersectionTriangulation::triangulate_intersection
(const MeshEntity &cell,
 const std::vector<double> &triangulation,
 std::size_t tri_tdim)
{
  // Compute the triangulation of the intersection of the cell and the
  // simplices of the flat triangulation vector with topology tdim.

  std::vector<double> total_triangulation;

  // Get dimensions (geometrical dimension assumed to be the same)
  const std::size_t cell_tdim = cell.mesh().topology().dim();
  const std::size_t gdim = cell.mesh().geometry().dim();

  // Store cell as std::vector<Point>
  // FIXME: Store as Point& ?
  std::vector<Point> simplex_cell(cell_tdim+1);
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  for (std::size_t j = 0; j < cell_tdim+1; ++j)
    simplex_cell[j] = geometry.point(vertices[j]);

  // Simplex in triangulation
  std::vector<Point> simplex(tri_tdim+1);
  const std::size_t offset = (tri_tdim+1)*gdim;

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size()/offset; ++i)
  {
    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < tri_tdim+1; ++j)
      for (std::size_t d = 0; d < gdim; ++d)
        simplex[j][d] = triangulation[offset*i+gdim*j+d];

    // Compute intersection
    std::vector<double> local_triangulation
      = triangulate_intersection(simplex_cell, cell_tdim,
                                 simplex, tri_tdim,
                                 gdim);

    // Add these to the net triangulation
    total_triangulation.insert(total_triangulation.end(),
                               local_triangulation.begin(),
                               local_triangulation.end());
  }

  return total_triangulation;
}
//-----------------------------------------------------------------------------
void
IntersectionTriangulation::triangulate_intersection
(const MeshEntity &cell,
 const std::vector<double> &triangulation,
 const std::vector<Point>& normals,
 std::vector<double>& intersection_triangulation,
 std::vector<Point>& intersection_normals,
 std::size_t tri_tdim)
{
  // Compute the triangulation of the intersection of the cell and the
  // simplices of the flat triangulation vector with topology tdim.

  // FIXME: clear or not?
  // intersection_triangulation.clear();
  // intersection_normals.clear();

  // Get dimensions (geometrical dimension assumed to be the same)
  const std::size_t cell_tdim = cell.mesh().topology().dim();
  const std::size_t gdim = cell.mesh().geometry().dim();

  // Store cell as std::vector<Point>
  // FIXME: Store as Point& ?
  std::vector<Point> simplex_cell(cell_tdim+1);
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  for (std::size_t j = 0; j < cell_tdim+1; ++j)
    simplex_cell[j] = geometry.point(vertices[j]);

  // Simplex in triangulation
  std::vector<Point> simplex(tri_tdim+1);
  const std::size_t offset = (tri_tdim+1)*gdim;

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size()/offset; ++i)
  {
    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < tri_tdim+1; ++j)
      for (std::size_t d = 0; d < gdim; ++d)
        simplex[j][d] = triangulation[offset*i+gdim*j+d];

    // Compute intersection
    std::vector<double> local_triangulation
      = triangulate_intersection(simplex_cell, cell_tdim,
                                 simplex, tri_tdim,
                                 gdim);

    // Add these to the net triangulation
    intersection_triangulation.insert(intersection_triangulation.end(),
                                      local_triangulation.begin(),
                                      local_triangulation.end());

    // Add the normal
    intersection_normals.resize(intersection_normals.size() + local_triangulation.size()/offset,
                                normals[i]);
  }

}


//------------------------------------------------------------------------------
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


//   std::vector<double> triangulation;

// #ifdef Augustdebug
//   std::cout << "before duplicates "<< points0.size() << '\n';
//   for (const auto p: points0)
//     std::cout << tools::matlabplot(p);
//   std::cout << '\n';
// #endif


//   // Remove duplicate points
//   std::vector<Point> points;
//   points.reserve(points0.size());

//   for (std::size_t i = 0; i < points0.size(); ++i)
//   {
//     bool different = true;
//     for (std::size_t j = i+1; j < points0.size(); ++j)
//       if ((points0[i] - points0[j]).norm() < DOLFIN_EPS)
//       {
// 	different = false;
// 	break;
//       }
//     if (different)
//       points.push_back(points0[i]);
//   }

//   if (points.size() < 3)
//   {
// #ifdef Augustdebug
//     std::cout << "after duplicate removal: " << points.size() << " too few points to form triangulation" << std::endl;
// #endif
//     return triangulation;
//   }

//   // Do the Graham scan starting at a suitable first node. Compute
//   // angles using a suitable reference line.
//   bool small_angle_diff = true;
//   std::vector<std::pair<double, std::size_t>> order;
//   std::size_t i_min = 0;

//   while (small_angle_diff and i_min < points.size())
//   {
//     order.clear();

//     // Find a suitable reference line (not too small)
//     std::size_t i_next;
//     Point ref;
//     bool small_length = true;
//     while (small_length and i_min < points.size())
//     {
//       i_next = (i_min + 1) % points.size();
//       ref = points[i_min] - points[i_next];
//       const double refnorm = ref.norm();
//       ref /= refnorm;
//       small_length = false;
//       if (refnorm < DOLFIN_EPS_LARGE)
//       {
// 	small_length = true;
// 	i_min++;
//       }
//     }

// #ifdef Augustdebug
//     std::cout << "i_min and next: " << i_min << ' '<<i_next << '\n';
//     std::cout << tools::matlabplot(points[i_min],"'ko'")<<tools::matlabplot(points[i_next],"'kx'")<<'\n';
// #endif

//     // Compute angles
//     for (std::size_t i = 0; i < points.size(); ++i)
//     {
//       if (i == i_min)
// 	continue;
//       const Point v = points[i] - points[i_min];
//       const double frac = ref.dot(v) / v.norm();
// #ifdef Augustdebug
//       std::cout << "frac = " << frac << '\n';
// #endif
//       double alpha;
//       if (frac <= -1)
// 	alpha = DOLFIN_PI;
//       else if (frac >= 1)
// 	alpha = 0;
//       else
// 	alpha = std::acos(frac);
//       order.push_back(std::make_pair(alpha, i));
//     }

//     // Sort points based on angle
//     std::sort(order.begin(), order.end());

//     // Compute angle differences. Select new point if angles are close.
//     small_angle_diff = false;
//     for (std::size_t i = 1; i < points.size(); ++i)
//       if (std::abs(order[i-1].first - order[i].first) < DOLFIN_EPS_LARGE)
//       {
// 	small_angle_diff = true;
// 	i_min++;
// 	break;
//       }
//   }

//   // If no points with small angles are found, return empty triangulation
//   if (small_angle_diff)
//     return triangulation;
//   //dolfin_assert(!small_angle_diff);
//   //dolfin_assert(i_min <= points.size());

// #ifdef Augustdebug
//   std::cout << "points\n";
//   for (const auto p: points)
//     std::cout << tools::matlabplot(p);
//   std::cout << '\n';
//   for (const auto o: order)
//     std::cout << o.second << ' ' << o.first*180/DOLFIN_PI<<'\n';
// #endif

//   // Triangulate polygon by connecting i_min with the ordered points
//   triangulation.reserve((points.size() - 2)*3*2);
//   const Point& p0 = points[i_min];
//   for (std::size_t i = 0; i < points.size() - 2; i++)
//   {
//     const Point& p1 = points[order[i].second];
//     const Point& p2 = points[order[i + 1].second];
//     triangulation.push_back(p0.x());
//     triangulation.push_back(p0.y());
//     triangulation.push_back(p1.x());
//     triangulation.push_back(p1.y());
//     triangulation.push_back(p2.x());
//     triangulation.push_back(p2.y());
//   }

//   return triangulation;
}
//-----------------------------------------------------------------------------
Point IntersectionTriangulation::intersection_edge_edge_2d(const Point& a,
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
  else
  {
    // no intersection, but return a point anyway. If we end up here
    // we have a conflict of interest between the predicate and
    // this function
    pt=0.25*(a+b+c+d);
  }

  CHECK_CGAL(pt, cgal_intersection_edge_edge_2d(a, b, c, d));

  // // Check if two edges are the same
  // const double same_point_tol = DOLFIN_EPS_LARGE;
  // if ((a - c).squared_norm() < same_point_tol and
  //     (b - d).squared_norm() < same_point_tol)
  //   return false;
  // if ((a - d).squared_norm() < same_point_tol and
  //     (b - c).norm() < same_point_tol)
  //   return false;

  // // Tolerance for orthogonality
  // const double orth_tol = DOLFIN_EPS_LARGE;

  // // Tolerance for coplanarity
  // //const double coplanar_tol = DOLFIN_EPS_LARGE;

  // const Point L1 = b - a;
  // const Point L2 = d - c;
  // const Point ca = c - a;
  // const Point n = L1.cross(L2);

  // // Check if L1 and L2 are coplanar (what if they're overlapping?)
  // // Update: always coplanar in 2D
  // // if (std::abs(ca.dot(n)) > coplanar_tol)
  // //   return false;

  // // Find orthogonal plane with normal n1
  // const Point n1 = n.cross(L1);
  // const double n1dotL2 = n1.dot(L2);

  // // If we have orthogonality
  // if (std::abs(n1dotL2) > orth_tol)
  // {
  //   const double t = -n1.dot(ca) / n1dotL2;

  //   // Find orthogonal plane with normal n2
  //   const Point n2 = n.cross(L2);
  //   const double n2dotL1 = n2.dot(L1);
  //   if (t >= 0 and
  //       t <= 1 and
  //       std::abs(n2dotL1) > orth_tol)
  //   {
  //     const double s = n2.dot(ca) / n2dotL1;
  //     if (s >= 0 and
  //         s <= 1)
  //     {
  // 	pt = a + s*L1;
  // 	return true;
  //     }
  //   }
  // }
  // // else // Now we have both coplanarity and colinearity, i.e. parallel lines
  // // {
  // // }

  // return false;

}
//-----------------------------------------------------------------------------
Point
IntersectionTriangulation::intersection_edge_edge(const Point& a,
						  const Point& b,
						  const Point& c,
						  const Point& d)
{
  dolfin_error("IntersectionTriangulation.cpp",
	       "intersection_edge_edge function",
	       "Not properly implemented");


  // // #ifdef Augustcgal
// //   dolfin_error("IntersectionTriangulation.cpp",
// // 	       "in intersection_edge_edge function",
// // 	       "cgal version only for 2d");
// //   return false;

// // #else

//   // // Check if two edges are the same
//   // const double same_point_tol = DOLFIN_EPS_LARGE;
//   // if ((a - c).squared_norm() < same_point_tol and
//   //     (b - d).squared_norm() < same_point_tol)
//   //   return false;
//   // if ((a - d).squared_norm() < same_point_tol and
//   //     (b - c).norm() < same_point_tol)
//   //   return false;

//   // Tolerance for orthogonality
//   const double orth_tol = DOLFIN_EPS_LARGE;

//   // Tolerance for coplanarity
//   const double coplanar_tol = DOLFIN_EPS_LARGE;

//   const Point L1 = b - a;
//   const Point L2 = d - c;
//   const Point ca = c - a;
//   const Point n = L1.cross(L2);

//   // // Check if L1 and L2 are coplanar (what if they're overlapping?)
//   // if (std::abs(ca.dot(n)) > coplanar_tol)
//   //   return false;

//   // Find orthogonal plane with normal n1
//   const Point n1 = n.cross(L1);
//   const double n1dotL2 = n1.dot(L2);

//   // If we have orthogonality
//   if (std::abs(n1dotL2) > orth_tol)
//   {
//     const double t = -n1.dot(ca) / n1dotL2;

//     // Find orthogonal plane with normal n2
//     const Point n2 = n.cross(L2);
//     const double n2dotL1 = n2.dot(L1);
//     if (t >= 0 and
//         t <= 1 and
//         std::abs(n2dotL1) > orth_tol)
//     {
//       const double s = n2.dot(ca) / n2dotL1;
//       if (s >= 0 and
//           s <= 1)
//       {
//   	pt = a + s*L1;
//   	//return true;
//       }
//     }
//   }
//   // else // Now we have both coplanarity and colinearity, i.e. parallel lines
//   // {
//   // }

//   return CHECK_CGAL(pt, cgal_intersection_edge_edge(a, b, c, d));
}
//-----------------------------------------------------------------------------
Point
IntersectionTriangulation::intersection_face_edge(const Point& r,
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

  // // Check if point is in triangle by calculating and checking
  // // barycentric coords.
  // const double d00 = rs.squared_norm();
  // const double d01 = rs.dot(rt);
  // const double d11 = rt.squared_norm();
  // const Point e2 = pt-r;
  // const double d20 = e2.dot(rs);
  // const double d21 = e2.dot(rt);
  // const double invdet = 1. / (d00*d11 - d01*d01);
  // const double v = (d11*d20 - d01*d21)*invdet;
  // // if (v < 0.)
  // //   return false;

  // const double w = (d00*d21 - d01*d20)*invdet;
  // if (w < 0.)
  //   return false;

  // if (v+w > 1.)
  //   return false;

  // return true;

  return CHECK_CGAL(pt, cgal_intersection_face_edge_2d(r, s, t, a, b));
}
//------------------------------------------------------------------------------
double IntersectionTriangulation::minimum_angle(double* a, double* b, double* c)
{
  // See Shewchuk: Lecture Notes on Geometric Robustness, April 15, 2013
  const double ab[2] = {a[0]-b[0], a[1]-b[1]};
  const double ac[2] = {a[0]-c[0], a[1]-c[1]};
  const double bc[2] = {b[0]-c[0], b[1]-c[1]};
  double l1 = std::sqrt(ab[0]*ab[0] + ab[1]*ab[1]);
  double l2 = std::sqrt(ac[0]*ac[0] + ac[1]*ac[1]);
  double l3 = std::sqrt(bc[0]*bc[0] + bc[1]*bc[1]);
  // Sort 3-way with l3 smallest
  if (l2 > l1) std::swap(l1, l2);
  if (l3 > l2) std::swap(l2, l3);
  if (l2 > l1) std::swap(l1, l2);
  const double sin_alpha = orient2d(a, b, c) / (l1 * l2);
  return asin(sin_alpha);
}
//------------------------------------------------------------------------------
