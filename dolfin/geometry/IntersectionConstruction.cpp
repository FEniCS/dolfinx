// Copyright (C) 2014-2016 Anders Logg, August Johansson and Benjamin Kehlet
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
// Last changed: 2017-02-08

#include <iomanip>
#include <dolfin/mesh/MeshEntity.h>
#include "predicates.h"
#include "CollisionPredicates.h"
#include "IntersectionConstruction.h"


// FIXME august
//#include <ttmath/ttmath.h>
// #include <Eigen/Dense>
// #include <algorithm>
//#define augustdebug

#ifdef augustdebug
#include "dolfin_simplex_tools.h"
#endif

namespace
{
  struct point_strictly_less
  {
    bool operator()(const dolfin::Point & p0, const dolfin::Point& p1)
    {
      if (p0.x() != p1.x())
	return p0.x() < p1.x();

      return p0.y() < p1.y();
    }
  };

  inline bool operator==(const dolfin::Point& p0, const dolfin::Point& p1)
  {
    return p0.x() == p1.x() && p0.y() == p1.y() && p0.z() == p1.z();
  }

  inline bool operator!=(const dolfin::Point& p0, const dolfin::Point& p1)
  {
    return p0.x() != p1.x() || p0.y() != p1.y() || p0.z() != p1.z();
  }

  inline bool operator<(const dolfin::Point& p0, const dolfin::Point& p1)
  {
    return p0.x() <= p1.x() && p0.y() <= p1.y() && p0.z() <= p1.z();
  }
}

using namespace dolfin;

//-----------------------------------------------------------------------------
// High-level intersection triangulation functions
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::intersection(const MeshEntity& entity_0,
                                       const MeshEntity& entity_1)
{
  // Get data
  const MeshGeometry& g0 = entity_0.mesh().geometry();
  const MeshGeometry& g1 = entity_1.mesh().geometry();
  const unsigned int* v0 = entity_0.entities(0);
  const unsigned int* v1 = entity_1.entities(0);

  // Pack data as vectors of points
  std::vector<Point> points_0(entity_0.dim() + 1);
  std::vector<Point> points_1(entity_1.dim() + 1);
  for (std::size_t i = 0; i <= entity_0.dim(); i++)
    points_0[i] = g0.point(v0[i]);
  for (std::size_t i = 0; i <= entity_1.dim(); i++)
    points_1[i] = g1.point(v1[i]);

  // Only look at first entity to get geometric dimension
  std::size_t gdim = g0.dim();

  // Call common implementation
  return intersection(points_0, points_1, gdim);
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::intersection(const std::vector<Point>& points_0,
                                       const std::vector<Point>& points_1,
                                       std::size_t gdim)
{
  // Get topological dimensions
  const std::size_t d0 = points_0.size() - 1;
  const std::size_t d1 = points_1.size() - 1;

  // Pick correct specialized implementation
  if (d0 == 1 && d1 == 1)
  {
    return intersection_segment_segment(points_0[0],
                                        points_0[1],
                                        points_1[0],
                                        points_1[1],
                                        gdim);
  }

  if (d0 == 2 && d1 == 1)
  {
    return intersection_triangle_segment(points_0[0],
					 points_0[1],
					 points_0[2],
					 points_1[0],
					 points_1[1],
					 gdim);
  }

  if (d0 == 1 && d1 == 2)
  {
    return intersection_triangle_segment(points_1[0],
                                         points_1[1],
                                         points_1[2],
                                         points_0[0],
                                         points_0[1],
                                         gdim);
  }

  if (d0 == 2 && d1 == 2)
    return intersection_triangle_triangle(points_0[0],
                                          points_0[1],
                                          points_0[2],
                                          points_1[0],
                                          points_1[1],
                                          points_1[2],
                                          gdim);

  if (d0 == 2 && d1 == 3)
    return intersection_tetrahedron_triangle(points_1[0],
                                             points_1[1],
                                             points_1[2],
                                             points_1[3],
                                             points_0[0],
                                             points_0[1],
                                             points_0[2]);

  if (d0 == 3 && d1 == 2)
    return intersection_tetrahedron_triangle(points_0[0],
                                             points_0[1],
                                             points_0[2],
                                             points_0[3],
                                             points_1[0],
                                             points_1[1],
                                             points_1[2]);

  if (d0 == 3 && d1 == 3)
    return intersection_tetrahedron_tetrahedron(points_0[0],
                                                points_0[1],
                                                points_0[2],
                                                points_0[3],
                                                points_1[0],
                                                points_1[1],
                                                points_1[2],
                                                points_1[3]);

  dolfin_error("IntersectionConstruction.cpp",
               "compute intersection",
               "Not implemented for dimensions %d / %d and geometric dimension %d", d0, d1, gdim);

  return std::vector<Point>();
}

//-----------------------------------------------------------------------------
// Low-level intersection triangulation functions
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::intersection_segment_segment(const Point& p0,
						       const Point& p1,
						       const Point& q0,
						       const Point& q1,
						       std::size_t gdim)
{
  switch (gdim)
  {
  case 1:
    {
      const std::vector<double> intersection_1d
	= intersection_segment_segment_1d(p0.x(), p1.x(), q0.x(), q1.x());
      std::vector<Point> intersection(intersection_1d.size());
      for (std::size_t i = 0; i < intersection.size(); ++i)
	intersection[i][0] = intersection_1d[i];
      return intersection;
    }
  case 2:
    return intersection_segment_segment_2d(p0, p1, q0, q1);
  case 3:
    return intersection_segment_segment_3d(p0, p1, q0, q1);
  default:
    dolfin_error("IntersectionConstruction.cpp",
		 "compute segment-segment collision",
		 "Unknown dimension %d.", gdim);
  }
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::intersection_triangle_segment(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& q0,
							const Point& q1,
							std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return intersection_triangle_segment_2d(p0, p1, p2, q0, q1);
  case 3:
    return intersection_triangle_segment_3d(p0, p1, p2, q0, q1);
  default:
    dolfin_error("IntersectionConstruction.cpp",
		 "compute triangle-segment intersection",
		 "Unknown dimension %d.", gdim);
  }
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::intersection_triangle_triangle(const Point& p0,
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
    return intersection_triangle_triangle_2d(p0, p1, p2, q0, q1, q2);
  case 3:
    return intersection_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);
  default:
    dolfin_error("IntersectionConstruction.cpp",
		 "compute segment-segment collision",
		 "Unknown dimension %d.", gdim);
  }
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
// Implementation of triangulation functions (private)
//-----------------------------------------------------------------------------
std::vector<double>
IntersectionConstruction::_intersection_segment_segment_1d(double p0,
							   double p1,
							   double q0,
							   double q1)
{
  std::vector<double> intersection;

  if (CollisionPredicates::collides_segment_segment_1d(p0, p1, q0, q1))
  {
    // Get range
    const double a0 = std::min(p0, p1);
    const double b0 = std::max(p0, p1);
    const double a1 = std::min(q0, q1);
    const double b1 = std::max(q0, q1);
    const double dx = std::min(b0 - a0, b1 - a1);
    intersection.resize(2);
    if (b0 - a1 < dx)
    {
      intersection[0] = a1;
      intersection[1] = b0;
    }
    else if (b1 - a0 < dx)
    {
      intersection[0] = a0;
      intersection[1] = b1;
    }
    else if (b0 - a0 < b1 - a1)
    {
      intersection[0] = a0;
      intersection[1] = b0;
    }
    else
    {
      intersection[0] = a1;
      intersection[1] = b1;
    }
  }

  return intersection;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_segment_segment_2d(Point p0,
							   Point p1,
							   Point q0,
							   Point q1)
{
#ifdef augustdebug
  std::cout << __FUNCTION__<<std::endl
  	    << tools::plot(p0)<<tools::plot(p1)<<tools::plot(q0)<<tools::plot(q1)<<tools::drawtriangle({p0,p1})<<tools::drawtriangle({q0,q1})<<std::endl;
#endif

  std::vector<Point> intersection;

  // Avoid some unnecessary computations
  if (!CollisionPredicates::collides_segment_segment_2d(p0, p1, q0, q1))
    return intersection;

  // Can we reduce to axis-aligned 1d?
  for (std::size_t d = 0; d < 2; ++d)
  {
    // Check if coordinates in dimension d is the same
    const bool reduce = (p0[d] == p1[d] and p1[d] == q0[d] and q0[d] == q1[d]);
    if (reduce)
    {
      const std::size_t j = (d+1) % 2;
      const std::vector<double> intersection_1d =
	intersection_segment_segment_1d(p0[j], p1[j], q0[j], q1[j]);
      intersection.resize(intersection_1d.size());
      for (std::size_t k = 0; k < intersection.size(); ++k)
      {
	intersection[k][d] = p0[d];
	intersection[k][j] = intersection_1d[k];
      }
      return intersection;
    }
  }

  intersection.reserve(4);

  // Check if the segment is actually a point
  if (p0 == p1)
  {
    if (CollisionPredicates::collides_segment_point_2d(q0, q1, p0))
    {
      intersection.push_back(p0);
      return intersection;
    }
  }

  if (q0 == q1)
  {
    if (CollisionPredicates::collides_segment_point_2d(p0, p1, q0))
    {
      intersection.push_back(q0);
      return intersection;
    }
  }

  // First test points to match procedure of
  // _collides_segment_segment_2d.
  if (CollisionPredicates::collides_segment_point_2d(q0, q1, p0))
  {
    intersection.push_back(p0);
  }
  if (CollisionPredicates::collides_segment_point_2d(q0, q1, p1))
  {
    intersection.push_back(p1);
  }
  if (CollisionPredicates::collides_segment_point_2d(p0, p1, q0))
  {
    intersection.push_back(q0);
  }
  if (CollisionPredicates::collides_segment_point_2d(p0, p1, q1))
  {
    intersection.push_back(q1);
  }

#ifdef augustdebug
  std::cout << " after point collisions: " <<intersection.size()<<" points: ";
  for (const Point p: intersection)
    std::cout << tools::plot(p);
  std::cout << std::endl;
#endif

  // Now we may have found all the intersectiosn
  if (intersection.size() == 1)
  {
    return intersection;
  }
  else if (intersection.size() > 1)
  {
    std::vector<Point> unique = unique_points(intersection);

    // assert that we only have one or two points
    dolfin_assert(intersection.size() == 2 ?
    		  (unique.size() == 1 or unique.size() == 2) :
    		  unique.size() == 2);
    return unique;
  }

  // The intersection is in principle given by p0 + numer / denom *
  // (p1 - p0), but we first identify certain cases where denom==0 and
  // / or numer == 0.

  // Use shortest distance as P0, P1
  const bool use_p = p0.squared_distance(p1) < q0.squared_distance(q1);
  Point P0, P1, Q0, Q1;
  if (use_p)
  {
    P0 = p0;
    P1 = p1;
    Q0 = q0;
    Q1 = q1;
  }
  else
  {
    P0 = q0;
    P1 = q1;
    Q0 = p0;
    Q1 = p1;
  }

  const double numer = orient2d(Q0.coordinates(), Q1.coordinates(), P0.coordinates());
  const double denom = (P1.x()-P0.x())*(Q1.y()-Q0.y()) - (P1.y()-P0.y())*(Q1.x()-Q0.x());

#ifdef augustdebug
  std::cout << "numer=" << numer << " denom="<<denom << " (equal? "<<(numer==denom) <<")\n";
#endif

  if (denom == 0. and numer == 0.)
  {
    // p0, p1 is collinear with q0, q1.
    if (p0.squared_distance(p1) < q0.squared_distance(q1))
    {
#ifdef augustdebug
      std::cout << "  swapped p0,p1,q0,q1\n";
#endif
      std::swap(p0, q0);
      std::swap(p1, q1);
    }
    const Point r = Q1 - p0;
    const double r2 = r.squared_norm();
    const Point rn = r / std::sqrt(r2);

    // FIXME: what to do if small?
    dolfin_assert(r2 > DOLFIN_EPS);

    double t0 = (Q0 - P0).dot(r) / r2;
    double t1 = (Q1 - P0).dot(r) / r2;
    if (t0 > t1)
    {
#ifdef augustdebug
      std::cout << "  swapped t0 and t1\n";
#endif
      std::swap(t0, t1);
    }

    if (CollisionPredicates::collides_segment_segment_1d(t0, t1, 0, 1))
    {
      // Compute two intersection points
      const Point z0 = P0 + std::max(0.0, t0)*r;
      const Point z1 = P0 + std::min(1.0, (Q0 - P0).dot(r) / r2 )*r;

#ifdef augustdebug
      std::cout << "case 1a"<<std::endl;
      std::cout.precision(22);
      std::cout <<"t0="<< t0<<"; t1="<<t1<<"; "<<tools::plot(p0)<<tools::plot(p1)<<" gave:\n"
      		<<tools::plot(z0,"'r.'")<<tools::plot(z1,"'g.'")<<std::endl;
#endif
      intersection.push_back(z0);
      intersection.push_back(z1);
    }
    else // Disjoint: no intersection
    {
#ifdef augustdebug
      std::cout << "case 1b"<<std::endl;
#endif
    }
  }
  else if (denom == 0. and numer != 0.)
  {
    // Parallel, disjoint
#ifdef augustdebug
    std::cout << "case 2"<<std::endl;
    std::cout << denom<<' '<<numer << std::endl;
#endif
  }
  else if (std::abs(denom) > DOLFIN_EPS_LARGE and
	   std::abs(std::abs(denom)-std::abs(numer)) > DOLFIN_EPS_LARGE)
  {
    Point x0 = P0 + numer / denom * (P1 - P0);

#ifdef augustdebug
    std::cout << "robust_linear_combination gave " << tools::plot(x0)<<std::endl;
#endif

    intersection.push_back(x0);
  }
  else // denom is small
  {
    // Now the line segments are almost collinear. Project on (1,0)
    // and (0,1) to find the largest projection. Note that Q0, Q1 is
    // longest.
    const std::size_t dim = (std::abs(Q0.x() - Q1.x()) > std::abs(Q0.y() - Q1.y())) ? 0 : 1;

    // Sort the points according to dim
    std::array<Point, 4> points = { P0, P1, Q0, Q1 };
    std::sort(points.begin(), points.end(), [dim](Point a,
						  Point b) {
		return a[dim] < b[dim];
	      });

    // Return midpoint
    Point xm = (points[1] + points[2]) / 2;
#ifdef augustdebug
    std::cout << "projection intersection with |denom|="<<std::abs(denom)<< " and difference "<< std::abs(std::abs(denom)-std::abs(numer)) <<" using P0, P1, Q0, Q1 " << tools::plot(P0)<<tools::plot(P1)<<tools::plot(Q0)<<tools::plot(Q1)<<"and dim = " << dim<< " gives\n" << tools::plot(xm)<<std::endl;
#endif
    intersection.push_back(xm);
  }

  const std::vector<Point> unique = unique_points(intersection);

#ifdef augustdebug
  std::cout << __FUNCTION__<< " gave unique points";
  std::cout << " (" << intersection.size()-unique.size()<< " duplicate pts found)\n";
  for (const Point p: unique)
    std::cout << tools::plot(p);
  std::cout << std::endl;
  // {
  //   std::cout << tools::generate_test(p0,p1,q0,q1,__FUNCTION__)<<std::endl;
  // }
#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
  std::cout<< " CGAL says: ";
  const std::vector<Point> cgal_intersection = cgal_intersection_segment_segment_2d(p0,p1,q0,q1);
  for (const Point p: cgal_intersection)
    std::cout <<std::setprecision(std::numeric_limits<long double>::digits10+2)  << tools::plot(p,"'gx','markersize',14")<<'\n';
#endif
#endif

  return unique;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_segment_segment_3d(const Point& p0,
							   const Point& p1,
							   const Point& q0,
							   const Point& q1)
{
  std::vector<Point> intersection;

  // Avoid some unnecessary computations
  if (!CollisionPredicates::collides_segment_segment_3d(p0, p1, q0, q1))
    return intersection;

  // FIXME: Can we reduce to 1d?

  intersection.reserve(4);

  // Check if the segment is actually a point
  if (p0 == p1)
  {
    if (CollisionPredicates::collides_segment_point_3d(q0, q1, p0))
    {
      intersection.push_back(p0);
      // std::cout << __FUNCTION__<<' '<<__LINE__<<std::endl;
      return intersection;
    }
  }

  if (q0 == q1)
  {
    if (CollisionPredicates::collides_segment_point_3d(p0, p1, q0))
    {
      intersection.push_back(q0);
      // std::cout << __FUNCTION__<<' '<<__LINE__<<std::endl;
      return intersection;
    }
  }

  // First test points to match procedure of
  // _collides_segment_segment_3d.
  if (CollisionPredicates::collides_segment_point_3d(q0, q1, p0))
  {
    intersection.push_back(p0);
  }
  if (CollisionPredicates::collides_segment_point_3d(q0, q1, p1))
  {
    intersection.push_back(p1);
  }
  if (CollisionPredicates::collides_segment_point_3d(p0, p1, q0))
  {
    intersection.push_back(q0);
  }
  if (CollisionPredicates::collides_segment_point_3d(p0, p1, q1))
  {
    intersection.push_back(q1);
  }

  // Now we may have found all the intersections
  if (intersection.size() == 1)
  {
    // std::cout << __FUNCTION__<<' '<<__LINE__<<std::endl;
    return intersection;
  }
  else if (intersection.size() > 1)
  {
    std::vector<Point> unique = unique_points(intersection);
    dolfin_assert(intersection.size() == 2 ?
    		  (unique.size() == 1 or unique.size() == 2) :
    		  unique.size() == 2);
    // std::cout << __FUNCTION__<<' '<<__LINE__<<std::endl;
    return unique;
  }

  // Follow Shewchuk Lecture Notes on Geometric Robustness
  const Point w = p0 - p1;
  const Point v = q0 - q1;
  const Point u = p1 - q1;
  const Point wv = w.cross(v);
  const Point vu = v.cross(u);
  const double denom = wv.squared_norm();
  const double numer = wv.dot(vu);

  if (denom == 0. and numer == 0)
  {
    //PPause;
  }
  else if (denom == 0 and numer != 0)
  {
    // Parallel, disjoint
    //PPause;
  }
  else if (denom != 0)
  {
    // Test Shewchuk

    // If fraction is close to 1, swap p0 and p1
    Point x0;

    if (std::abs(numer / denom - 1) < DOLFIN_EPS_LARGE)
    {
      const Point u_swapped = p0 - q1;
      const Point vu_swapped = v.cross(u_swapped);
      const double numer_swapped = -wv.dot(vu_swapped);
      x0 = p0 + numer_swapped / denom * (p1 - p0);
      // std::cout << __FUNCTION__<<' '<<__LINE__<<std::endl;
    }
    else
    {
      // std::cout << __FUNCTION__<<' '<<__LINE__<<std::endl;
      x0 = p1 + numer / denom * (p0 - p1);
    }

    intersection.push_back(x0);
  }

  std::vector<Point> unique = unique_points(intersection);
  return unique;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_triangle_segment_2d(const Point& p0,
							    const Point& p1,
							    const Point& p2,
							    const Point& q0,
							    const Point& q1)
{
  std::vector<Point> points;

  if (CollisionPredicates::collides_triangle_point_2d(p0, p1, p2, q0))
    points.push_back(q0);
  if (CollisionPredicates::collides_triangle_point_2d(p0, p1, p2, q1))
    points.push_back(q1);

  if (CollisionPredicates::collides_segment_segment_2d(p0, p1, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_2d(p0, p1, q0, q1);
    // FIXME: Should we require consistency between collision and intersection
    //dolfin_assert(intersection.size());
    points.insert(points.end(), intersection.begin(), intersection.end());
  }

  if (CollisionPredicates::collides_segment_segment_2d(p0, p2, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_2d(p0, p2, q0, q1);
    //dolfin_assert(intersection.size());
    points.insert(points.end(), intersection.begin(), intersection.end());
  }

  if (CollisionPredicates::collides_segment_segment_2d(p1, p2, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_2d(p1, p2, q0, q1);
    //dolfin_assert(intersection.size());
    points.insert(points.end(), intersection.begin(), intersection.end());
  }

  // Remove strict duplictes. Use exact equality here. Approximate
  // equality is for ConvexTriangulation.
  // FIXME: This can be avoided if we use interior segment tests.
  const std::vector<Point> unique = unique_points(points);
  return unique;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_triangle_segment_3d(Point p0,
							    Point p1,
							    Point p2,
							    Point q0,
							    Point q1)
{
#ifdef augustdebug
  std::cout << "-----------------\n" << __FUNCTION__<<'\n';
  std::cout << tools::drawtriangle({p0,p1,p2})<<tools::drawtriangle({q0,q1})<<'\n';
#endif

  std::vector<Point> points;

  if (CollisionPredicates::collides_triangle_point_3d(p0, p1, p2, q0))
    points.push_back(q0);

  // std::cout <<__FUNCTION__<<' '<<__LINE__<<'\n';
  // for (const Point p: points)
  //   std::cout << tools::plot3(p);
  // std::cout << std::endl;

  if (CollisionPredicates::collides_triangle_point_3d(p0, p1, p2, q1))
    points.push_back(q1);

  // std::cout <<__FUNCTION__<<' '<<__LINE__<<'\n';
  // for (const Point p: points)
  //   std::cout << tools::plot3(p);
  // std::cout << std::endl;

  // We may finish
  if (points.size() == 2) return points;

  if (CollisionPredicates::collides_segment_segment_3d(p0, p1, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_3d(p0, p1, q0, q1);
    // FIXME: Should we require consistency between collision and intersection
    //dolfin_assert(intersection.size());
    points.insert(points.end(), intersection.begin(), intersection.end());
  }
  // std::cout <<__FUNCTION__<<' '<<__LINE__<<'\n';
  // for (const Point p: points)
  //   std::cout << tools::plot3(p);
  // std::cout << std::endl;

  if (CollisionPredicates::collides_segment_segment_3d(p0, p2, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_3d(p0, p2, q0, q1);
    //dolfin_assert(intersection.size());
    points.insert(points.end(), intersection.begin(), intersection.end());
  }
  // std::cout <<__FUNCTION__<<' '<<__LINE__<<'\n';
  // for (const Point p: points)
  //   std::cout << tools::plot3(p);
  // std::cout << std::endl;

  if (CollisionPredicates::collides_segment_segment_3d(p1, p2, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_3d(p1, p2, q0, q1);
    //dolfin_assert(intersection.size());
    points.insert(points.end(), intersection.begin(), intersection.end());
  }
  // std::cout <<__FUNCTION__<<' '<<__LINE__<<'\n';
  // for (const Point p: points)
  //   std::cout << tools::plot3(p);
  // std::cout << std::endl;


  // Think ray-plane intersection if segment is not in plane (which it
  // shouldn't be)

  // normal
  Point n = cross_product(p0,p1,p2);
  n /= n.norm();

  // Eqn for plane is n.x = d
  const double d = n.dot(p0);

  // Ray is r(t) = q0 + t(q1 - q0). Intersection with plane is at t =
  // (d - n.dot(q0)) / n.dot(q1-q0). Make sure normal is not
  // orthogonal to plane (should be detected above though)

  const Point x = q0 + (d - n.dot(q0)) / n.dot(q1-q0) * (q1 - q0);
  std::cout << CollisionPredicates::collides_triangle_point_3d(p0,p1,p2, x)<<'\n';

  points.push_back(x);




  // // Compute intersection of segment and plane and check if point
  // // found is inside triangle. Let the plane be defined by p0, p1,
  // // p2. The intersection point is then given by p = a + z(b-a) where
  // // z = orient3d(a,d,e,c) / det(a-b, d-c, e-c)

  // // Let a = q0, d = p0, e = p1, c = p2
  // const double numer = orient3d(q0.coordinates(), p0.coordinates(), p1.coordinates(), p2.coordinates());
  // const double denom = det(q0 - q1, p0 - p2, p1 - p2);

  // if (denom == 0 and numer != 0)
  // {
  //   // line is parallel to plane
  //   std::cout << numer << ' '<< denom << "    "<<CollisionPredicates::collides_triangle_segment_3d(p0,p1,p2, q0,q1)<<'\n';
  //   PPause;
  // }
  // else if (denom == 0 and numer == 0)
  // {
  //   // line lies in the the plane: this should be taken care of above
  //   // since numer == 0 means q0 in p0, p1, p2

  //   PPause;
  // }
  // else
  // {
  //   Point x0;

  //   // If fraction is close to 1, swap q0 and q1
  //   if (std::abs(numer / denom - 1) < DOLFIN_EPS_LARGE)
  //   {
  //     const double numer_swapped = orient3d(q1.coordinates(), p0.coordinates(), p1.coordinates(), p2.coordinates());
  //     // Denominator flips sign
  //     x0 = q1 - numer_swapped / denom * (q0 - q1);
  //   }
  //   else
  //   {
  //     x0 = q0 + numer / denom * (q1 - q0);
  //   }
  //   points.push_back(x0);
  // }


  // Remove strict duplictes. Use exact equality here. Approximate
  // equality is for ConvexTriangulation.
  // FIXME: This can be avoided if we use interior segment tests.
  const std::vector<Point> unique = unique_points(points);

  // {
  //   std::cout << __FUNCTION__<<' '<<tools::drawtriangle({p0,p1,p2})<<tools::drawtriangle({q0,q1})<<"   ";
  //   for (const Point p: unique)
  //     std::cout << tools::plot3(p);
  //   std::cout << std::endl;
  // }

  return unique;
}

//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_triangle_triangle_2d(Point p0,
							     Point p1,
							     Point p2,
							     Point q0,
							     Point q1,
							     Point q2)
{
#ifdef augustdebug
  std::cout << __FUNCTION__<<" intersection of\n"
  	    << tools::drawtriangle({p0,p1,p2})<<tools::drawtriangle({q0,q1,q2})<<std::endl;
#endif

  std::vector<Point> points;

  if (CollisionPredicates::collides_triangle_triangle_2d(p0, p1, p2,
  							 q0, q1, q2))
  {
    // Pack points as vectors
    std::array<Point, 3> tri_0({p0, p1, p2});
    std::array<Point, 3> tri_1({q0, q1, q2});

    // Find all vertex-vertex collision
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = 0; j < 3; j++)
      {
  	if (tri_0[i] == tri_1[j])
  	  points.push_back(tri_0[i]);
      }
    }

#ifdef augustdebug
    std::cout << " after vertex--vertex collisions: total " << points.size() <<" points: ";
    for (const Point p: points) std::cout << tools::plot3(p);
    std::cout << std::endl;
#endif

    // Find all vertex-"edge interior" intersections
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = 0; j < 3; j++)
      {
  	if (tri_0[i] != tri_1[j] && tri_0[(i+1)%3] != tri_1[j] &&
  	    CollisionPredicates::collides_segment_point_2d(tri_0[i], tri_0[(i+1)%3], tri_1[j]))
  	  points.push_back(tri_1[j]);

  	if (tri_1[i] != tri_0[j] && tri_1[(i+1)%3] != tri_0[j] &&
  	    CollisionPredicates::collides_segment_point_2d(tri_1[i], tri_1[(i+1)%3], tri_0[j]))
  	  points.push_back(tri_0[j]);
      }
    }

#ifdef augustdebug
    std::cout << " after vertex edge interior collisions: total " << points.size() <<" points: ";
    for (const Point p: points) std::cout << tools::plot3(p);
    std::cout << std::endl;
#endif

    // Find all "edge interior"-"edge interior" intersections
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = 0; j < 3; j++)
      {
  	{
  	  std::vector<Point> triangulation =
  	    intersection_segment_segment_2d(tri_0[i],
					    tri_0[(i+1)%3],
					    tri_1[j],
					    tri_1[(j+1)%3]);

  	  points.insert(points.end(), triangulation.begin(), triangulation.end());
  	}
      }
    }

#ifdef augustdebug
    std::cout << " after edge interior -- edge interior collisions: total " << points.size() <<" points: ";
    for (const Point p: points) std::cout << tools::plot(p);
    std::cout << std::endl;
#endif

    // Find alle vertex-"triangle interior" intersections
    const int s0 = std::signbit(orient2d(tri_0[0].coordinates(), tri_0[1].coordinates(), tri_0[2].coordinates())) == true ? -1 : 1;
    const int s1 = std::signbit(orient2d(tri_1[0].coordinates(), tri_1[1].coordinates(), tri_1[2].coordinates())) == true ? -1 : 1;

    for (std::size_t i = 0; i < 3; ++i)
    {
      const double q0_q1_pi = s1*orient2d(tri_1[0].coordinates(), tri_1[1].coordinates(), tri_0[i].coordinates());
      const double q1_q2_pi = s1*orient2d(tri_1[1].coordinates(), tri_1[2].coordinates(), tri_0[i].coordinates());
      const double q2_q0_pi = s1*orient2d(tri_1[2].coordinates(), tri_1[0].coordinates(), tri_0[i].coordinates());

      if (q0_q1_pi > 0. and
  	  q1_q2_pi > 0. and
  	  q2_q0_pi > 0.)
      {
  	points.push_back(tri_0[i]);
      }

      const double p0_p1_qi = s0*orient2d(tri_0[0].coordinates(), tri_0[1].coordinates(), tri_1[i].coordinates());
      const double p1_p2_qi = s0*orient2d(tri_0[1].coordinates(), tri_0[2].coordinates(), tri_1[i].coordinates());
      const double p2_p0_qi = s0*orient2d(tri_0[2].coordinates(), tri_0[0].coordinates(), tri_1[i].coordinates());

      if (p0_p1_qi > 0. and
  	  p1_p2_qi > 0. and
  	  p2_p0_qi > 0.)
      {
  	points.push_back(tri_1[i]);
      }
    }

#ifdef augustdebug
    std::cout << " after vertex -- triangle collisions: total " << points.size() <<" points: ";
    for (const Point p: points) std::cout << tools::plot(p);
    std::cout << std::endl;
#endif
  }


  return unique_points(points);
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_triangle_triangle_3d(const Point& p0,
							     const Point& p1,
							     const Point& p2,
							     const Point& q0,
							     const Point& q1,
							     const Point& q2)
{
  dolfin_error("IntersectionConstruction.cpp",
	       "compute triangle-triangle 3d intersection",
	       "Not implemented.");
  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_tetrahedron_triangle(const Point& p0,
							     const Point& p1,
							     const Point& p2,
							     const Point& p3,
							     const Point& q0,
							     const Point& q1,
							     const Point& q2)
{
  // This code mimics the triangulate_tetrahedron_tetrahedron and the
  // triangulate_tetrahedron_tetrahedron_triangle_codes: we first
  // identify triangle nodes in the tetrahedra. Them we continue with
  // edge-face detection for the four faces of the tetrahedron and the
  // triangle. The points found are used to form a triangulation by
  // first sorting them using a Graham scan.

  // Pack points as vectors
  std::array<Point, 4> tet = {{p0, p1, p2, p3}};
  std::array<Point, 3> tri = {{q0, q1, q2}};

  // Tolerance for duplicate points (p and q are the same if
  // (p-q).norm() < same_point_tol)
  const double same_point_tol = DOLFIN_EPS;

  // Tolerance for small triangle (could be improved by identifying
  // sliver and small triangles)
  const double tri_det_tol = DOLFIN_EPS;

  std::vector<Point> points;

  // Triangle node in tetrahedron intersection
  for (std::size_t i = 0; i < 3; ++i)
    if (CollisionPredicates::collides_tetrahedron_point(tet[0],
							tet[1],
							tet[2],
							tet[3],
							tri[i]))
      points.push_back(tri[i]);

  // Check if a tetrahedron edge intersects the triangle
  const std::array<std::array<std::size_t, 2>, 6> tet_edges = {{ {2, 3},
								 {1, 3},
								 {1, 2},
								 {0, 3},
								 {0, 2},
								 {0, 1} }};
  for (std::size_t e = 0; e < 6; ++e)
    if (CollisionPredicates::collides_triangle_segment_3d(tri[0], tri[1], tri[2],
							  tet[tet_edges[e][0]],
							  tet[tet_edges[e][1]]))
    {
      const std::vector<Point> ii = intersection_triangle_segment_3d(tri[0], tri[1], tri[2],
                                                                     tet[tet_edges[e][0]],
                                                                     tet[tet_edges[e][1]]);
#ifdef augustdebug
      std::cout << __FUNCTION__<<'\n'
		<< tools::drawtriangle({p0,p1,p2,p3})<<tools::drawtriangle({q0,q1,q2})<<'\n'
		<< tools::drawtriangle({tri[0], tri[1], tri[2]})<<tools::drawtriangle({tet[tet_edges[e][0]],tet[tet_edges[e][1]]})<<'\n';
#endif

      dolfin_assert(ii.size());
      points.insert(points.end(), ii.begin(), ii.end());
    }

  // check if a triangle edge intersects a tetrahedron face
  const std::array<std::array<std::size_t, 3>, 4> tet_faces = {{ {1, 2, 3},
								 {0, 2, 3},
								 {0, 1, 3},
								 {0, 1, 2} }};
  const std::array<std::array<std::size_t, 2>, 3> tri_edges = {{ {0, 1},
								 {0, 2},
								 {1, 2} }};
  for (std::size_t f = 0; f < 4; ++f)
    for (std::size_t e = 0; e < 3; ++e)
      if (CollisionPredicates::collides_triangle_segment_3d(tet[tet_faces[f][0]],
							    tet[tet_faces[f][1]],
							    tet[tet_faces[f][2]],
							    tri[tri_edges[e][0]],
							    tri[tri_edges[e][1]]))
      {
	const std::vector<Point> ii = intersection_triangle_segment_3d(tet[tet_faces[f][0]],
                                                                       tet[tet_faces[f][1]],
                                                                       tet[tet_faces[f][2]],
                                                                       tri[tri_edges[e][0]],
                                                                       tri[tri_edges[e][1]]);
	dolfin_assert(ii.size());
	points.insert(points.end(), ii.begin(), ii.end());
      }

  // FIXME: segment-segment intersection should not be needed if
  // triangle-segment intersection doesn't miss this

  // Remove duplicate nodes
  // FIXME: If this is necessary, reuse code from ConvexTriangulation
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

  return points;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_tetrahedron_tetrahedron(const Point& p0,
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
  const std::array<Point, 4> tet_0 = {p0, p1, p2, p3};
  const std::array<Point, 4> tet_1 = {q0, q1, q2, q3};

  // Points in the triangulation (unique)
  std::vector<Point> points;

  // Node intersection
  for (int i = 0; i < 4; ++i)
  {
    if (CollisionPredicates::collides_tetrahedron_point(tet_0[0],
							tet_0[1],
							tet_0[2],
							tet_0[3],
							tet_1[i]))
      points.push_back(tet_1[i]);

    if (CollisionPredicates::collides_tetrahedron_point(tet_1[0],
							tet_1[1],
							tet_1[2],
							tet_1[3],
							tet_0[i]))
      points.push_back(tet_0[i]);
  }

  // Edge face intersections
  const std::array<std::array<std::size_t, 2>, 6> edges = {{ {2, 3},
							     {1, 3},
							     {1, 2},
							     {0, 3},
							     {0, 2},
							     {0, 1} }};
  const std::array<std::array<std::size_t, 3>, 4> faces = {{ {1, 2, 3},
							     {0, 2, 3},
							     {0, 1, 3},
							     {0, 1, 2} }};
  // Loop over edges e and faces f
  for (std::size_t e = 0; e < 6; ++e)
  {
    for (std::size_t f = 0; f < 4; ++f)
    {
      if (CollisionPredicates::collides_triangle_segment_3d(tet_0[faces[f][0]],
							    tet_0[faces[f][1]],
							    tet_0[faces[f][2]],
							    tet_1[edges[e][0]],
							    tet_1[edges[e][1]]))
      {
	const std::vector<Point> intersection
	  = intersection_triangle_segment_3d(tet_0[faces[f][0]],
					     tet_0[faces[f][1]],
					     tet_0[faces[f][2]],
					     tet_1[edges[e][0]],
					     tet_1[edges[e][1]]);
	points.insert(points.end(), intersection.begin(), intersection.end());
      }

      if (CollisionPredicates::collides_triangle_segment_3d(tet_1[faces[f][0]],
							    tet_1[faces[f][1]],
							    tet_1[faces[f][2]],
							    tet_0[edges[e][0]],
							    tet_0[edges[e][1]]))
      {
	const std::vector<Point> intersection
	  = intersection_triangle_segment_3d(tet_1[faces[f][0]],
					     tet_1[faces[f][1]],
					     tet_1[faces[f][2]],
					     tet_0[edges[e][0]],
					     tet_0[edges[e][1]]);
	points.insert(points.end(), intersection.begin(), intersection.end());
      }
    }
  }

  // Edge edge intersection
  Point pt;
  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      if (CollisionPredicates::collides_segment_segment_3d(tet_0[edges[i][0]],
							   tet_0[edges[i][1]],
							   tet_1[edges[j][0]],
							   tet_1[edges[j][1]]))
      {
	const std::vector<Point> intersection
	  = intersection_segment_segment_3d(tet_0[edges[i][0]],
					    tet_0[edges[i][1]],
					    tet_1[edges[j][0]],
					    tet_1[edges[j][1]]);
	points.insert(points.end(), intersection.begin(), intersection.end());
      }
    }
  }

  std::vector<Point> unique = unique_points(points);
  return unique;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::unique_points(std::vector<Point> input_points)
{
  std::vector<Point> unique;
  unique.reserve(input_points.size());

  for (std::size_t i = 0; i < input_points.size(); ++i)
  {
    bool found = false;
    for (std::size_t j = i+1; j < input_points.size(); ++j)
      if (input_points[i] == input_points[j])
      {
	found = true;
	break;
      }
    if (!found)
      unique.push_back(input_points[i]);
  }

  return unique;
}

//-----------------------------------------------------------------------------
 double IntersectionConstruction::det(Point ab, Point dc, Point ec)

{
  double a = ab.x(), b = ab.y(), c = ab.z();
  double d = dc.x(), e = dc.y(), f = dc.z();
  double g = ec.x(), h = ec.y(), i = ec.z();
  return a * (e * i - f * h)
       + b * (f * g - d * i)
       + c * (d * h - e * g);
}
//-----------------------------------------------------------------------------
Point IntersectionConstruction::cross_product(Point a,
					      Point b,
					      Point c)
{
  // Accurate cross product p = (a-c) x (b-c). See Shewchuk Lecture
  // Notes on Geometric Robustness.
  double ayz[2] = {a.y(), a.z()};
  double byz[2] = {b.y(), b.z()};
  double cyz[2] = {c.y(), c.z()};
  double azx[2] = {a.z(), a.x()};
  double bzx[2] = {b.z(), b.x()};
  double czx[2] = {c.z(), c.x()};
  double axy[2] = {a.x(), a.y()};
  double bxy[2] = {b.x(), b.y()};
  double cxy[2] = {c.x(), c.y()};
  Point p(orient2d(ayz, byz, cyz),
	  orient2d(azx, bzx, czx),
	  orient2d(axy, bxy, cxy));
  return p;
}
//-----------------------------------------------------------------------------
