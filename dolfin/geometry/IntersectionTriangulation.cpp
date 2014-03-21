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
// Last changed: 2014-03-21

#include <dolfin/mesh/MeshEntity.h>
#include "IntersectionTriangulation.h"
#include "CollisionDetection.h"

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
      dolfin_not_implemented();
      break;
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

  dolfin_not_implemented();
  std::vector<double> triangulation;
  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_triangle_triangle
(const MeshEntity& c0, const MeshEntity& c1)
{
  // This algorithm computes the (convex) polygon resulting from the
  // intersection of two triangles. It then triangulates the polygon
  // by trivially drawing an edge from one vertex to all other
  // vertices. The polygon is computed by first identifying all
  // vertex-cell collisions and then all edge-edge collisions. The
  // points are then sorted using a simplified Graham scan (simplified
  // since we know the polygon is convex).

  dolfin_assert(c0.mesh().topology().dim() == 2);
  dolfin_assert(c1.mesh().topology().dim() == 2);

  // // Tolerance for duplicate points (p and q are the same if
  // // (p-q).norm() < same_point_tol)
  // const double same_point_tol = DOLFIN_EPS_LARGE;

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

  // // Create empty list of collision points
  // std::vector<Point> points;

  // // Find all vertex-cell collisions
  // for (std::size_t i = 0; i < 3; i++)
  // {
  //   const Point p0 = geometry_0.point(vertices_0[i]);
  //   const Point p1 = geometry_1.point(vertices_1[i]);
  //   if (CollisionDetection::collides_triangle_point(c1, p0))
  //     points.push_back(p0);
  //   if (CollisionDetection::collides_triangle_point(c0, p1))
  //     points.push_back(p1);
  // }

  // // Find all edge-edge collisions (not needed?)
  // for (std::size_t i0 = 0; i0 < 3; i0++)
  // {
  //   const std::size_t j0 = (i0 + 1) % 3;
  //   const Point p0 = geometry_0.point(vertices_0[i0]);
  //   const Point q0 = geometry_0.point(vertices_0[j0]);
  //   for (std::size_t i1 = 0; i1 < 3; i1++)
  //   {
  //     const std::size_t j1 = (i1 + 1) % 3;
  //     const Point p1 = geometry_1.point(vertices_1[i1]);
  //     const Point q1 = geometry_1.point(vertices_1[j1]);
  //     Point point;
  //     if (intersection_edge_edge(p0, q0, p1, q1, point))
  //       points.push_back(point);
  //   }
  // }

  // // Remove duplicate points
  // std::vector<Point> tmp;
  // tmp.reserve(points.size());

  // for (std::size_t i = 0; i < points.size(); ++i)
  // {
  //   bool different = true;
  //   for (std::size_t j = i+1; j < points.size(); ++j)
  //     if ((points[i] - points[j]).norm() < same_point_tol)
  //     {
  //       different = false;
  //       break;
  //     }
  //   if (different)
  //     tmp.push_back(points[i]);
  // }
  // points = tmp;

  // // Special case: no points found
  // std::vector<double> triangulation;
  // if (points.size() < 3)
  //   return triangulation;

  // // Find left-most point (smallest x-coordinate)
  // std::size_t i_min = 0;
  // double x_min = points[0].x();
  // for (std::size_t i = 1; i < points.size(); i++)
  // {
  //   const double x = points[i].x();
  //   if (x < x_min)
  //   {
  //     x_min = x;
  //     i_min = i;
  //   }
  // }

  // // Compute signed squared cos of angle with (0, 1) from i_min to all points
  // std::vector<std::pair<double, std::size_t> > order;
  // for (std::size_t i = 0; i < points.size(); i++)
  // {
  //   // Skip left-most point used as origin
  //   if (i == i_min)
  //     continue;

  //   // Compute vector to point
  //   const Point v = points[i] - points[i_min];

  //   // Compute square cos of angle
  //   const double cos2 = (v.y() < 0.0 ? -1.0 : 1.0)*v.y()*v.y() / v.squared_norm();

  //   // Store for sorting
  //   order.push_back(std::make_pair(cos2, i));
  // }

  // // Sort points based on angle
  // std::sort(order.begin(), order.end());

  // // Triangulate polygon by connecting i_min with the ordered points
  // triangulation.reserve((points.size() - 2)*3*2);
  // const Point& p0 = points[i_min];
  // for (std::size_t i = 0; i < points.size() - 2; i++)
  // {
  //   const Point& p1 = points[order[i].second];
  //   const Point& p2 = points[order[i + 1].second];
  //   triangulation.push_back(p0.x());
  //   triangulation.push_back(p0.y());
  //   triangulation.push_back(p1.x());
  //   triangulation.push_back(p1.y());
  //   triangulation.push_back(p2.x());
  //   triangulation.push_back(p2.y());
  // }

  // return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_triangle
(const MeshEntity& tetrahedron, const MeshEntity& triangle)
{
  // This code mimics the
  // triangulate_intersection_tetrahedron_tetrahedron and the
  // triangulate_intersection_tetrahedron_tetrahedron_triangle_codes:
  // we first identify triangle nodes in the tetrahedra. Them we
  // continue with edge-face detection for the four faces of the
  // tetrahedron and the triangle. The points found are used to form a
  // triangulation by first sorting them using a Graham scan.

  dolfin_assert(tetrahedron.mesh().topology().dim() == 3);
  dolfin_assert(triangle.mesh().topology().dim() == 2);

  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  const double same_point_tol = DOLFIN_EPS_LARGE;

  // Tolerance for small triangle (could be improved by identifying
  // sliver and small triangles)
  const double tri_det_tol = DOLFIN_EPS_LARGE;

  // Get the vertices as points
  const MeshGeometry& tet_geom = tetrahedron.mesh().geometry();
  const unsigned int* tet_vert = tetrahedron.entities(0);

  const MeshGeometry& tri_geom = triangle.mesh().geometry();
  const unsigned int* tri_vert = triangle.entities(0);
  const Point q0 = tri_geom.point(tri_vert[0]);
  const Point q1 = tri_geom.point(tri_vert[1]);
  const Point q2 = tri_geom.point(tri_vert[2]);

  std::vector<Point> points;

  // Triangle node in tetrahedron intersection
  if (CollisionDetection::collides_tetrahedron_point(tetrahedron, q0))
    points.push_back(q0);

  if (CollisionDetection::collides_tetrahedron_point(tetrahedron, q1))
    points.push_back(q1);

  if (CollisionDetection::collides_tetrahedron_point(tetrahedron, q2))
    points.push_back(q2);

  // Check if a tetrahedron edge intersects the triangle
  std::vector<std::vector<int> > tet_edges(6, std::vector<int>(2));
  tet_edges[0][0] = tet_vert[2];
  tet_edges[0][1] = tet_vert[3];
  tet_edges[1][0] = tet_vert[1];
  tet_edges[1][1] = tet_vert[3];
  tet_edges[2][0] = tet_vert[1];
  tet_edges[2][1] = tet_vert[2];
  tet_edges[3][0] = tet_vert[0];
  tet_edges[3][1] = tet_vert[3];
  tet_edges[4][0] = tet_vert[0];
  tet_edges[4][1] = tet_vert[2];
  tet_edges[5][0] = tet_vert[0];
  tet_edges[5][1] = tet_vert[1];

  Point pt;
  for (std::size_t e = 0; e < 6; ++e)
  {
    if (intersection_face_edge(q0, q1, q2,
			       tet_geom.point(tet_edges[e][0]),
			       tet_geom.point(tet_edges[e][1]),
			       pt))
      points.push_back(pt);
  }

  // Check if a triangle edge intersects a tetrahedron face
  std::vector<std::vector<std::size_t> >
    tet_faces(4, std::vector<std::size_t>(3));

  tet_faces[0][0] = tet_vert[1];
  tet_faces[0][1] = tet_vert[2];
  tet_faces[0][2] = tet_vert[3];
  tet_faces[1][0] = tet_vert[0];
  tet_faces[1][1] = tet_vert[2];
  tet_faces[1][2] = tet_vert[3];
  tet_faces[2][0] = tet_vert[0];
  tet_faces[2][1] = tet_vert[1];
  tet_faces[2][2] = tet_vert[3];
  tet_faces[3][0] = tet_vert[0];
  tet_faces[3][1] = tet_vert[1];
  tet_faces[3][2] = tet_vert[2];

  for (std::size_t f = 0; f < 4; ++f)
  {
    if (intersection_face_edge(tet_geom.point(tet_faces[f][0]),
			       tet_geom.point(tet_faces[f][1]),
			       tet_geom.point(tet_faces[f][2]),
			       q0, q1, pt))
      points.push_back(pt);
    if (intersection_face_edge(tet_geom.point(tet_faces[f][0]),
			       tet_geom.point(tet_faces[f][1]),
			       tet_geom.point(tet_faces[f][2]),
			       q0, q2, pt))
      points.push_back(pt);
    if (intersection_face_edge(tet_geom.point(tet_faces[f][0]),
			       tet_geom.point(tet_faces[f][1]),
			       tet_geom.point(tet_faces[f][2]),
			       q1, q2, pt))
      points.push_back(pt);
  }


  // edge edge intersection
  for (std::size_t f = 0; f < 6; ++f)
  {
    if (intersection_edge_edge(tet_geom.point(tet_edges[f][0]),
			       tet_geom.point(tet_edges[f][1]),
			       q0, q1,
			       pt))
      points.push_back(pt);
    if (intersection_edge_edge(tet_geom.point(tet_edges[f][0]),
			       tet_geom.point(tet_edges[f][1]),
			       q0,q2,
			       pt))
      points.push_back(pt);
    if (intersection_edge_edge(tet_geom.point(tet_edges[f][0]),
			       tet_geom.point(tet_edges[f][1]),
			       q1,q2,
			       pt))
      points.push_back(pt);
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
    if (std::abs(det) > tri_det_tol)
    {
      if (det < -tri_det_tol)
	std::swap(points[0], points[1]);

      // One triangle with three vertices in 3D gives 9 doubles
      triangulation.resize(9);
      for (std::size_t m = 0, idx = 0; m < 3; ++m)
	for (std::size_t d = 0; d < 3; ++d, ++idx)
	  triangulation[idx] = points[m][d];
    }
    return triangulation;
  }

  // Tesselate as in the triangle-triangle intesection case: First
  // sort points using a Graham scan, then connect to form triangles.

  // Use the center of the points and point no 0 as reference for the
  // angle calculation
  Point pointscenter = points[0];
  for (std::size_t m = 1; m < points.size(); ++m)
    pointscenter += points[m];
  pointscenter /= points.size();

  std::vector<std::pair<double, std::size_t> > order;
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

  // Tesselate
  for (std::size_t m = 0; m < order.size()-1; ++m)
  {
    // Candidate triangle
    std::vector<Point> cand(3);
    cand[0] = points[0];
    cand[1] = points[order[m].second];
    cand[2] = points[order[m + 1].second];

    // Include triangle if determinant is sufficiently large
    const double det = ((cand[2] - cand[1]).cross(cand[1] - cand[0])).norm();
    if (std::abs(det) > tri_det_tol)
    {
      if (det < -tri_det_tol)
	std::swap(cand[0], cand[1]);
      for (std::size_t n = 0; n < 3; ++n)
	for (std::size_t d = 0; d < 3; ++d)
	  triangulation.push_back(cand[n][d]);
    }
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_tetrahedron
(const MeshEntity& tetrahedron_0,
 const MeshEntity& tetrahedron_1)
{
  // This algorithm computes the intersection of cell_0 and cell_1 by
  // returning a vector<double> with points describing a tetrahedral
  // mesh of the intersection. We will use the fact that the
  // intersection is a convex polyhedron. The algorithm works by first
  // identifying intersection points: vertex points inside a cell,
  // edge-face collision points and edge-edge collision points (the
  // edge-edge is a rare occurance). Having the intersection points,
  // we identify points that are coplanar and thus form a facet of the
  // polyhedron. These points are then used to form a tesselation of
  // triangles, which are used to form tetrahedra by the use of the
  // center point of the polyhedron. This center point is thus an
  // additional point not found on the polyhedron facets.

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

  // // Tolerance for coplanar points
  // const double coplanar_tol = 1000*DOLFIN_EPS_LARGE;

  // // Tolerance for the tetrahedron determinant (otherwise problems
  // // with warped tets)
  // const double tet_det_tol = DOLFIN_EPS_LARGE;

  // // Tolerance for duplicate points (p and q are the same if
  // // (p-q).norm() < same_point_tol)
  // const double same_point_tol = DOLFIN_EPS_LARGE;

  // // Tolerance for small triangle (could be improved by identifying
  // // sliver and small triangles)
  // const double tri_det_tol = DOLFIN_EPS_LARGE;

  // // Points in the triangulation (unique)
  // std::vector<Point> points;

  // // Get the vertices as points
  // const MeshGeometry& geom_0 = tetrahedron_0.mesh().geometry();
  // const unsigned int* vert_0 = tetrahedron_0.entities(0);
  // const MeshGeometry& geom_1 = tetrahedron_1.mesh().geometry();
  // const unsigned int* vert_1 = tetrahedron_1.entities(0);

  // // Node intersection
  // for (int i = 0; i<4; ++i)
  // {
  //   if (CollisionDetection::collides_tetrahedron_point(tetrahedron_0,
  //       					       geom_1.point(vert_1[i])))
  //     points.push_back(geom_1.point(vert_1[i]));

  //   if (CollisionDetection::collides_tetrahedron_point(tetrahedron_1,
  //       					       geom_0.point(vert_0[i])))
  //     points.push_back(geom_0.point(vert_0[i]));
  // }

  // // Edge face intersections
  // std::vector<std::vector<std::size_t> > edges_0(6, std::vector<std::size_t>(2));
  // edges_0[0][0] = vert_0[2];
  // edges_0[0][1] = vert_0[3];
  // edges_0[1][0] = vert_0[1];
  // edges_0[1][1] = vert_0[3];
  // edges_0[2][0] = vert_0[1];
  // edges_0[2][1] = vert_0[2];
  // edges_0[3][0] = vert_0[0];
  // edges_0[3][1] = vert_0[3];
  // edges_0[4][0] = vert_0[0];
  // edges_0[4][1] = vert_0[2];
  // edges_0[5][0] = vert_0[0];
  // edges_0[5][1] = vert_0[1];

  // std::vector<std::vector<std::size_t> > edges_1(6, std::vector<std::size_t>(2));
  // edges_1[0][0] = vert_1[2];
  // edges_1[0][1] = vert_1[3];
  // edges_1[1][0] = vert_1[1];
  // edges_1[1][1] = vert_1[3];
  // edges_1[2][0] = vert_1[1];
  // edges_1[2][1] = vert_1[2];
  // edges_1[3][0] = vert_1[0];
  // edges_1[3][1] = vert_1[3];
  // edges_1[4][0] = vert_1[0];
  // edges_1[4][1] = vert_1[2];
  // edges_1[5][0] = vert_1[0];
  // edges_1[5][1] = vert_1[1];

  // std::vector<std::vector<std::size_t> > faces_0(4, std::vector<std::size_t>(3));
  // faces_0[0][0] = vert_0[1];
  // faces_0[0][1] = vert_0[2];
  // faces_0[0][2] = vert_0[3];
  // faces_0[1][0] = vert_0[0];
  // faces_0[1][1] = vert_0[2];
  // faces_0[1][2] = vert_0[3];
  // faces_0[2][0] = vert_0[0];
  // faces_0[2][1] = vert_0[1];
  // faces_0[2][2] = vert_0[3];
  // faces_0[3][0] = vert_0[0];
  // faces_0[3][1] = vert_0[1];
  // faces_0[3][2] = vert_0[2];

  // std::vector<std::vector<std::size_t> > faces_1(4, std::vector<std::size_t>(3));
  // faces_1[0][0] = vert_1[1];
  // faces_1[0][1] = vert_1[2];
  // faces_1[0][2] = vert_1[3];
  // faces_1[1][0] = vert_1[0];
  // faces_1[1][1] = vert_1[2];
  // faces_1[1][2] = vert_1[3];
  // faces_1[2][0] = vert_1[0];
  // faces_1[2][1] = vert_1[1];
  // faces_1[2][2] = vert_1[3];
  // faces_1[3][0] = vert_1[0];
  // faces_1[3][1] = vert_1[1];
  // faces_1[3][2] = vert_1[2];

  // // Loop over edges e and faces f
  // for (std::size_t e = 0; e < 6; ++e)
  //   for (std::size_t f = 0; f < 4; ++f)
  //   {
  //     Point pta;
  //     if (intersection_face_edge(geom_0.point(faces_0[f][0]),
  //       			 geom_0.point(faces_0[f][1]),
  //       			 geom_0.point(faces_0[f][2]),
  //       			 geom_1.point(edges_1[e][0]),
  //       			 geom_1.point(edges_1[e][1]),
  //       			 pta))
  // 	points.push_back(pta);

  //     Point ptb;
  //     if (intersection_face_edge(geom_1.point(faces_1[f][0]),
  //       			 geom_1.point(faces_1[f][1]),
  //       			 geom_1.point(faces_1[f][2]),
  //       			 geom_0.point(edges_0[e][0]),
  //       			 geom_0.point(edges_0[e][1]),
  //       			 ptb))
  // 	points.push_back(ptb);
  //   }

  // // Edge edge intersection
  // for (int i = 0; i < 6; ++i)
  //   for (int j = 0; j < 6; ++j)
  //   {
  //     Point pt;
  //     if (intersection_edge_edge(geom_0.point(edges_0[i][0]),
  //       			 geom_0.point(edges_0[i][1]),
  //       			 geom_1.point(edges_1[j][0]),
  //       			 geom_1.point(edges_1[j][1]),
  //       			 pt))
  // 	points.push_back(pt);
  //   }

  // // Remove duplicate nodes
  // std::vector<Point> tmp;
  // tmp.reserve(points.size());
  // for (std::size_t i = 0; i < points.size(); ++i)
  // {
  //   bool different=true;
  //   for (std::size_t j = i+1; j < points.size(); ++j)
  //   {
  //     if ((points[i] - points[j]).norm() < same_point_tol) {
  // 	different = false;
  // 	break;
  //     }
  //   }

  //   if (different)
  //     tmp.push_back(points[i]);
  // }
  // points = tmp;

  // // We didn't find sufficiently many points: can't form any
  // // tetrahedra.
  // if (points.size() < 4)
  //   return std::vector<double>();

  // // Points forming the tetrahedral partitioning of the polyhedron. We
  // // have 4 points per tetrahedron in three dimensions => 12 doubles
  // // per tetrahedron.
  // std::vector<double> triangulation;

  // // Start forming a tesselation
  // if (points.size() == 4)
  // {
  //   // Include if determinant is sufficiently large. The determinant
  //   // can possibly be computed in a more stable way if needed.
  //   const double det = (points[3] - points[0]).dot
  //     ((points[1] - points[0]).cross(points[2] - points[0]));

  //   if (std::abs(det) > tet_det_tol)
  //   {
  //     if (det < -tet_det_tol)
  //       std::swap(points[0], points[1]);

  //     // One tet with four vertices in 3D gives 12 doubles
  //     triangulation.resize(12);
  //     for (std::size_t m = 0, idx = 0; m < 4; ++m)
  // 	for (std::size_t d = 0; d < 3; ++d, ++idx)
  // 	  triangulation[idx] = points[m][d];
  //   }
  //   // Note: this can be empty if the tetrahedron was not sufficiently
  //   // large
  //   return triangulation;
  // }

  // // Tetrahedra are created using the facet points and a center point.
  // Point polyhedroncenter = points[0];
  // for (std::size_t i = 1; i < points.size(); ++i)
  //   polyhedroncenter += points[i];
  // polyhedroncenter /= points.size();

  // // Data structure for storing checked triangle indices (do this
  // // better with some fancy stl structure?)
  // const std::size_t N = points.size(), N2 = points.size()*points.size();
  // std::vector<int> checked(N*N2 + N2 + N, 0);

  // // Find coplanar points
  // for (std::size_t i = 0; i < points.size(); ++i)
  //   for (std::size_t j = i+1; j < points.size(); ++j)
  //     for (std::size_t k = 0; k < points.size(); ++k)
  // 	if (checked[i*N2 + j*N + k] == 0 and k != i and k != j)
  // 	{
  // 	  // Check that triangle area is sufficiently large
  // 	  Point n = (points[j] - points[i]).cross(points[k] - points[i]);
  // 	  const double tridet = n.norm();
  // 	  if (tridet < tri_det_tol)
  //           break;

  // 	  // Normalize normal
  // 	  n /= tridet;

  // 	  // Compute triangle center
  // 	  const Point tricenter = (points[i] + points[j] + points[k]) / 3.;

  // 	  // Check whether all other points are on one side of thus
  // 	  // facet. Initialize as true for the case of only three
  // 	  // coplanar points.
  // 	  bool on_convex_hull = true;

  // 	  // Compute dot products to check which side of the plane
  // 	  // (i,j,k) we're on. Note: it seems to be better to compute
  // 	  // n.dot(points[m]-n.dot(tricenter) rather than
  // 	  // n.dot(points[m]-tricenter).
  // 	  std::vector<double> ip(points.size(), -(n.dot(tricenter)));
  // 	  for (std::size_t m = 0; m < points.size(); ++m)
  // 	    ip[m] += n.dot(points[m]);

  // 	  // Check inner products range by finding max & min (this
  // 	  // seemed possibly more numerically stable than checking all
  // 	  // vs all and then break).
  // 	  double minip = 9e99, maxip = -9e99;
  // 	  for (size_t m = 0; m < points.size(); ++m)
  // 	    if (m != i and m != j and m != k)
  // 	    {
  // 	      minip = (minip > ip[m]) ? ip[m] : minip;
  // 	      maxip = (maxip < ip[m]) ? ip[m] : maxip;
  // 	    }

  // 	  // Different sign => triangle is not on the convex hull
  // 	  if (minip*maxip < -DOLFIN_EPS)
  // 	    on_convex_hull = false;

  // 	  if (on_convex_hull)
  // 	  {
  // 	    // Find all coplanar points on this facet given the
  // 	    // tolerance coplanar_tol
  // 	    std::vector<std::size_t> coplanar;
  // 	    for (std::size_t m = 0; m < points.size(); ++m)
  // 	      if (std::abs(ip[m]) < coplanar_tol)
  // 		coplanar.push_back(m);

  // 	    // Mark this plane (how to do this better?)
  // 	    for (std::size_t m = 0; m < coplanar.size(); ++m)
  // 	      for (std::size_t n = m+1; n < coplanar.size(); ++n)
  // 		for (std::size_t o = n+1; o < coplanar.size(); ++o)
  // 		  checked[coplanar[m]*N2 + coplanar[n]*N + coplanar[o]]
  //                   = checked[coplanar[m]*N2 + coplanar[o]*N + coplanar[n]]
  // 		    = checked[coplanar[n]*N2 + coplanar[m]*N + coplanar[o]]
  // 		    = checked[coplanar[n]*N2 + coplanar[o]*N + coplanar[m]]
  // 		    = checked[coplanar[o]*N2 + coplanar[n]*N + coplanar[m]]
  // 		    = checked[coplanar[o]*N2 + coplanar[m]*N + coplanar[n]]
  //                   = 1;

  // 	    // Do the actual tesselation using the coplanar points and
  // 	    // a center point
  // 	    if (coplanar.size() == 3)
  // 	    {
  // 	      // Form one tetrahedron
  // 	      std::vector<Point> cand(4);
  // 	      cand[0] = points[coplanar[0]];
  // 	      cand[1] = points[coplanar[1]];
  // 	      cand[2] = points[coplanar[2]];
  // 	      cand[3] = polyhedroncenter;

  // 	      // Include if determinant is sufficiently large
  // 	      const double det = (cand[3]-cand[0]).dot
  //               ((cand[1] - cand[0]).cross(cand[2] - cand[0]));
  // 	      if (std::abs(det) > tet_det_tol)
  // 	      {
  // 		if (det < -tet_det_tol)
  // 		  std::swap(cand[0], cand[1]);

  // 		for (std::size_t m = 0; m < 4; ++m)
  // 		  for (std::size_t d = 0; d < 3; ++d)
  // 		    triangulation.push_back(cand[m][d]);
  // 	      }

  // 	    }
  // 	    else if (coplanar.size() > 3)
  // 	    {
  // 	      // Tesselate as in the triangle-triangle intersection
  // 	      // case: First sort points using a Graham scan, then
  // 	      // connect to form triangles. Finally form tetrahedra
  // 	      // using the center of the polyhedron.

  // 	      // Use the center of the coplanar points and point no 0
  // 	      // as reference for the angle calculation
  // 	      Point pointscenter = points[coplanar[0]];
  // 	      for (std::size_t m = 1; m < coplanar.size(); ++m)
  // 		pointscenter += points[coplanar[m]];
  // 	      pointscenter /= coplanar.size();

  // 	      std::vector<std::pair<double, std::size_t> > order;
  // 	      Point ref = points[coplanar[0]] - pointscenter;
  // 	      ref /= ref.norm();

  // 	      // Calculate and store angles
  // 	      for (std::size_t m = 1; m < coplanar.size(); ++m)
  // 	      {
  // 		const Point v = points[coplanar[m]] - pointscenter;
  // 		const double frac = ref.dot(v) / v.norm();
  // 		double alpha;
  // 		if (frac <= -1)
  //                 alpha=DOLFIN_PI;
  // 		else if (frac>=1)
  //                 alpha=0;
  // 		else
  //               {
  // 		  alpha = acos(frac);
  // 		  if (v.dot(n.cross(ref)) < 0)
  //                   alpha = 2*DOLFIN_PI-alpha;
  // 		}
  // 		order.push_back(std::make_pair(alpha, m));
  // 	      }

  // 	      // Sort angles
  // 	      std::sort(order.begin(), order.end());

  // 	      // Tesselate
  // 	      for (std::size_t m = 0; m < coplanar.size()-2; ++m)
  // 	      {
  // 		// Candidate tetrahedron:
  // 		std::vector<Point> cand(4);
  // 		cand[0] = points[coplanar[0]];
  // 		cand[1] = points[coplanar[order[m].second]];
  // 		cand[2] = points[coplanar[order[m + 1].second]];
  // 		cand[3] = polyhedroncenter;

  // 		// Include tetrahedron if determinant is "large"
  // 		const double det = (cand[3] - cand[0]).dot
  //                 ((cand[1] - cand[0]).cross(cand[2] - cand[0]));
  // 		if (std::abs(det) > tet_det_tol)
  // 		{
  // 		  if (det < -tet_det_tol)
  // 		    std::swap(cand[0], cand[1]);
  // 		  for (std::size_t n = 0; n < 4; ++n)
  // 		    for (std::size_t d = 0; d < 3; ++d)
  // 		      triangulation.push_back(cand[n][d]);
  // 		}
  // 	      }
  // 	    }
  // 	  }
  // 	}


  // return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection
(const MeshEntity &cell,
 const std::vector<double> &triangulation)
{
  std::vector<double> total_triangulation;

  // Get dimensions
  const std::size_t tdim = cell.mesh().topology().dim();
  const std::size_t gdim = cell.mesh().geometry().dim();
  const std::size_t no_nodes = tdim+1;
  const std::size_t offset = no_nodes*gdim;

  std::vector<Point> simplex_cell(no_nodes), simplex(no_nodes);
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);

  // Loop over all simplices
  for (std::size_t i = 0; i < triangulation.size()/offset; ++i)
  {

    // Store simplices as std::vector<Point>
    for (std::size_t j = 0; j < no_nodes; ++j)
    {
      simplex_cell[j] = geometry.point(vertices[j]);

      for (std::size_t d = 0; d < gdim; ++d)
        simplex[j][d] = triangulation[offset*i+gdim*j+d];
    }

    // Compute intersection triangulation
    std::vector<double> local_triangulation;
    switch(tdim) {
    case 2:
      local_triangulation = IntersectionTriangulation::triangulate_intersection_triangle_triangle(simplex_cell, simplex);
      break;
    case 3:
      local_triangulation = IntersectionTriangulation::triangulate_intersection_tetrahedron_tetrahedron(simplex_cell, simplex);
      break;
    default:
      dolfin_error("IntersectionTriangulation.cpp",
                   "triangulate intersection of cell and triangulation array",
                   "unknown dimension of triangulation array");
    }

    // Add these to the net triangulation
    total_triangulation.insert(total_triangulation.end(),
                               local_triangulation.begin(),
                               local_triangulation.end());
  }

  return total_triangulation;
}
//-----------------------------------------------------------------------------
bool
IntersectionTriangulation::intersection_edge_edge(const Point& a,
						  const Point& b,
						  const Point& c,
						  const Point& d,
						  Point& pt)
{
  // Check if two edges are the same
  const double same_point_tol = DOLFIN_EPS_LARGE;
  if ((a - c).norm() < same_point_tol and
      (b - d).norm() < same_point_tol)
    return false;
  if ((a - d).norm() < same_point_tol and
      (b - c).norm() < same_point_tol)
    return false;

  // Tolerance for orthogonality
  const double orth_tol = DOLFIN_EPS_LARGE;

  // Tolerance for coplanarity
  const double coplanar_tol = DOLFIN_EPS_LARGE;

  const Point L1 = b - a;
  const Point L2 = d - c;
  const Point ca = c - a;
  const Point n = L1.cross(L2);

  // Check if L1 and L2 are coplanar (what if they're overlapping?)
  if (std::abs(ca.dot(n)) > coplanar_tol)
    return false;

  // Find orthogonal plane with normal n1
  const Point n1 = n.cross(L1);
  const double n1dotL2 = n1.dot(L2);

  // If we have orthogonality
  if (std::abs(n1dotL2) > orth_tol)
  {
    const double t = n1.dot(a - c) / n1dotL2;

    // Find orthogonal plane with normal n2
    const Point n2 = n.cross(L2);
    const double n2dotL1 = n2.dot(L1);
    if (t >= 0 and
        t <= 1 and
        std::abs(n2dotL1) > orth_tol)
    {
      const double s = n2.dot(c - a) / n2dotL1;
      if (s >= 0 and
          s <= 1)
      {
	pt = a + s*L1;
	return true;
      }
    }
  }
  return false;
}
//-----------------------------------------------------------------------------
bool
IntersectionTriangulation::intersection_face_edge(const Point& r,
						  const Point& s,
						  const Point& t,
						  const Point& a,
						  const Point& b,
						  Point& pt)
{
  // This standard edge face intersection test is as follows:
  // - Check if end points of the edge (a,b) on opposite side of plane
  // given by the face (r,s,t)
  // - If we have sign change, compute intersection with plane.
  // - Check if computed point is on triangle given by face.

  // If the edge and the face are in the same plane, we return false
  // and leave this to the edge-edge intersection test.

  // Tolerance for edga and face in plane (topologically 2D problem)
  const double top_2d_tol = DOLFIN_EPS_LARGE;

  // Compute normal
  const Point rs = s - r;
  const Point rt = t - r;
  Point n = rs.cross(rt);
  n /= n.norm();

  // Check sign change (note that if either dot product is zero it's
  // orthogonal)
  const double da = n.dot(a - r);
  const double db = n.dot(b - r);

  // Note: if da and db we may have edge intersection (detected in
  // other routine)
  if (da*db > 0)
    return false;

  // Face and edge are in topological 2d: taken care of in edge-edge
  // intersection or point in simplex.
  const double sum = std::abs(da) + std::abs(db);
  if (sum < top_2d_tol)
    return false;

  // Calculate intersection
  pt = a + std::abs(da) / sum * (b - a);

  // Check if point is in triangle by calculating and checking
  // barycentric coords.
  const double d00 = rs.squared_norm();
  const double d01 = rs.dot(rt);
  const double d11 = rt.squared_norm();
  const Point e2 = pt-r;
  const double d20 = e2.dot(rs);
  const double d21 = e2.dot(rt);
  const double invdet = 1. / (d00*d11 - d01*d01);
  const double v = (d11*d20 - d01*d21)*invdet;
  if (v < 0.)
    return false;

  const double w = (d00*d21 - d01*d20)*invdet;
  if (w < 0.)
    return false;

  if (v+w > 1.)
    return false;

  return true;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_triangle_triangle
(const std::vector<Point>& tri_0,
 const std::vector<Point>& tri_1)
{
  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  const double same_point_tol = DOLFIN_EPS_LARGE;


  // Create empty list of collision points
  std::vector<Point> points;

  // Find all vertex-cell collisions
  for (std::size_t i = 0; i < 3; i++)
  {
    const Point p0 = tri_0[i];//geometry_0.point(vertices_0[i]);
    const Point p1 = tri_1[i]; //geometry_1.point(vertices_1[i]);
    // Note: this routine is changed to being public:
    if (CollisionDetection::collides_triangle_point(tri_1[0],
                                                    tri_1[1],
                                                    tri_1[2],
                                                    p0))
      points.push_back(p0);
    if (CollisionDetection::collides_triangle_point(tri_0[0],
                                                    tri_0[1],
                                                    tri_0[2],
                                                    p1))
      points.push_back(p1);
  }

  // Find all edge-edge collisions (not needed?)
  for (std::size_t i0 = 0; i0 < 3; i0++)
  {
    const std::size_t j0 = (i0 + 1) % 3;
    const Point p0 = tri_0[i0]; //geometry_0.point(vertices_0[i0]);
    const Point q0 = tri_0[j0]; //geometry_0.point(vertices_0[j0]);
    for (std::size_t i1 = 0; i1 < 3; i1++)
    {
      const std::size_t j1 = (i1 + 1) % 3;
      const Point p1 = tri_1[i1];//geometry_1.point(vertices_1[i1]);
      const Point q1 = tri_1[j1];//geometry_1.point(vertices_1[j1]);
      Point point;
      if (intersection_edge_edge(p0, q0, p1, q1, point))
        points.push_back(point);
    }
  }

  // Remove duplicate points
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

  // Special case: no points found
  std::vector<double> triangulation;
  if (points.size() < 3)
    return triangulation;

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
  std::vector<std::pair<double, std::size_t> > order;
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

  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionTriangulation::triangulate_intersection_tetrahedron_tetrahedron
(const std::vector<Point>& tet_0,
 const std::vector<Point>& tet_1)
{
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
  std::vector<std::vector<std::size_t> > edges_0(6, std::vector<std::size_t>(2));
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

  std::vector<std::vector<std::size_t> > edges_1(6, std::vector<std::size_t>(2));
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

  std::vector<std::vector<std::size_t> > faces_0(4, std::vector<std::size_t>(3));
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

  std::vector<std::vector<std::size_t> > faces_1(4, std::vector<std::size_t>(3));
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
      Point pta;
      if (intersection_face_edge(tet_0[faces_0[f][0]],
				 tet_0[faces_0[f][1]],
				 tet_0[faces_0[f][2]],
				 tet_1[edges_1[e][0]],
				 tet_1[edges_1[e][1]],
				 pta))
  	points.push_back(pta);

      Point ptb;
      if (intersection_face_edge(tet_1[faces_1[f][0]],
				 tet_1[faces_1[f][1]],
				 tet_1[faces_1[f][2]],
				 tet_0[edges_0[e][0]],
				 tet_0[edges_0[e][1]],
				 ptb))
  	points.push_back(ptb);
    }

  // Edge edge intersection
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
    {
      Point pt;
      if (intersection_edge_edge(tet_0[edges_0[i][0]],
				 tet_0[edges_0[i][1]],
				 tet_1[edges_1[j][0]],
				 tet_1[edges_1[j][1]],
				 pt))
  	points.push_back(pt);
    }

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

  // Start forming a tesselation
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
  std::vector<int> checked(N*N2 + N2 + N, 0);

  // Find coplanar points
  for (std::size_t i = 0; i < points.size(); ++i)
    for (std::size_t j = i+1; j < points.size(); ++j)
      for (std::size_t k = 0; k < points.size(); ++k)
  	if (checked[i*N2 + j*N + k] == 0 and k != i and k != j)
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
  	  std::vector<double> ip(points.size(), -(n.dot(tricenter)));
  	  for (std::size_t m = 0; m < points.size(); ++m)
  	    ip[m] += n.dot(points[m]);

  	  // Check inner products range by finding max & min (this
  	  // seemed possibly more numerically stable than checking all
  	  // vs all and then break).
  	  double minip = 9e99, maxip = -9e99;
  	  for (size_t m = 0; m < points.size(); ++m)
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
  	    for (std::size_t m = 0; m < points.size(); ++m)
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
                    = 1;

  	    // Do the actual tesselation using the coplanar points and
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
  	      // Tesselate as in the triangle-triangle intersection
  	      // case: First sort points using a Graham scan, then
  	      // connect to form triangles. Finally form tetrahedra
  	      // using the center of the polyhedron.

  	      // Use the center of the coplanar points and point no 0
  	      // as reference for the angle calculation
  	      Point pointscenter = points[coplanar[0]];
  	      for (std::size_t m = 1; m < coplanar.size(); ++m)
  		pointscenter += points[coplanar[m]];
  	      pointscenter /= coplanar.size();

  	      std::vector<std::pair<double, std::size_t> > order;
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

  	      // Tesselate
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
