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
// Last changed: 2017-01-05

#include <dolfin/mesh/MeshEntity.h>
#include "predicates.h"
#include "CollisionPredicates.h"
#include "IntersectionConstruction.h"


// FIXME august
#include <ttmath/ttmath.h>
#include </home/august/dolfin_simplex_tools.h>
#include <Eigen/Dense>
#include <algorithm>
// #define augustdebug


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

  if (d0 == 2 && d1 == 2)
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
	= intersection_segment_segment_1d(p0[0], p1[0], q0[0], q1[0]);
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

  // Can we reduce to 1d?
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

  if (intersection.size() == 1)
  {
    return intersection;
  }
  else if (intersection.size() > 1)
  {
    std::vector<Point> unique = unique_points(intersection);
    dolfin_assert(intersection.size() == 2 ?
    		  (unique.size() == 1 or unique.size() == 2) :
    		  unique.size() == 2);
    return unique;
  }

  const double denom = (p1.x()-p0.x())*(q1.y()-q0.y()) - (p1.y()-p0.y())*(q1.x()-q0.x());
  const double numer = orient2d(q0.coordinates(), q1.coordinates(), p0.coordinates());

  if (denom == 0. and numer == 0.)
  {
    // p0, p1 is collinear with q0, q1.
    // Take the longest distance as p0, p1
    if (p0.squared_distance(p1) < q0.squared_distance(q1))
    {
#ifdef augustdebug
      std::cout << "  swapped p0,p1,q0,q1\n";
#endif
      std::swap(p0, q0);
      std::swap(p1, q1);
    }
    const Point r = p1 - p0;
    const double r2 = r.squared_norm();
    const Point rn = r / std::sqrt(r2);

    // FIXME: what to do if small?
    dolfin_assert(r2 > DOLFIN_EPS);

    double t0 = (q0 - p0).dot(r) / r2;
    double t1 = (q1 - p0).dot(r) / r2;
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
      const Point z0 = p0 + std::max(0., t0)*r;
      const Point z1 = p0 + std::min(1.,(q0 - p0).dot(r) / r2 )*r;

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
    // {
    //   std::cout << "case 2 with TT\n";
    //   typedef ttmath::Big<TTMATH_BITS(22), TTMATH_BITS(104)> TT;
    //   const TT ax(a.x()), ay(a.y()), bx(b.x()), by(b.y()), cx(c.x()), cy(c.y()), dx(d.x()), dy(d.y());
    //   const TT numer_tt(numer);
    //   const TT denom_tt = (bx-ax)*(dy-cy) - (by-ay)*(dx-cx);
    //   const TT u_tt = numer_tt / denom_tt;
    //   const TT zx = cx + u_tt*(dx - cx);
    //   const TT zy = cy + u_tt*(dy - cy);
    //   std::cout << " numer_tt / denom_tt = u_tt = " << u_tt << std::endl
    // 		<< " plot("<<zx<<','<<zy<<",'mx','markersize',18);"<<std::endl;

    // }
  }
  else if (denom != 0.)
  {
    // Test Shewchuk
    const Point x0 = p0 + numer / denom * (p1 - p0);
  #ifdef augustdebug
      std::cout << "  test shewchuk p0+numer/denom*(p1-p0): "<<tools::plot(x0)<<'\n';
#endif

    if ((CollisionPredicates::collides_segment_point_1d(p0[0], p1[0], x0[0]) and
	 CollisionPredicates::collides_segment_point_1d(p0[1], p1[1], x0[1]) and
	 CollisionPredicates::collides_segment_point_1d(q0[0], q1[0], x0[0]) and
	 CollisionPredicates::collides_segment_point_1d(q0[1], q1[1], x0[1])) or
	(CollisionPredicates::collides_segment_point_2d(p0, p1, x0) and
	 CollisionPredicates::collides_segment_point_2d(q0, q1, x0)))
    {
#ifdef augustdebug
      std::cout << "  shewchuk std gave: "<<tools::plot(x0)<<'\n';
      for (std::size_t d = 0; d < 2; ++d)
	std::cout << CollisionPredicates::collides_segment_point_1d(p0[d],p1[d],x0[d])<<' ';
      std::cout << '\n';
      for (std::size_t d = 0; d < 2; ++d)
	std::cout << CollisionPredicates::collides_segment_point_1d(q0[d],q1[d],x0[d])<<' ';
      std::cout << '\n';
      std::cout << CollisionPredicates::collides_segment_point_2d(p0, p1, x0)<<' '<<CollisionPredicates::collides_segment_point_2d(q0, q1, x0)<<'\n';
#endif
      intersection.push_back(x0);
    }
    else // test Shewchuk with points swapped
    {
      const double denom_q = (q1.x()-q0.x())*(p1.y()-p0.y()) - (q1.y()-q0.y())*(p1.x()-p0.x());
#ifdef augustdebug
      std::cout << " checking swapped " << orient2d(p0.coordinates(), p1.coordinates(), q0.coordinates())<<' '<<denom_q <<'\n';
#endif
      dolfin_assert(denom_q != 0); // this should have been taken care of

      const Point x1 = q0 + orient2d(p0.coordinates(), p1.coordinates(), q0.coordinates()) /  denom_q * (q1 - q0);
      if ((CollisionPredicates::collides_segment_point_1d(p0[0], p1[0], x1[0]) and
	   CollisionPredicates::collides_segment_point_1d(p0[1], p1[1], x1[1]) and
	   CollisionPredicates::collides_segment_point_1d(q0[0], q1[0], x1[0]) and
	   CollisionPredicates::collides_segment_point_1d(q0[1], q1[1], x1[1])) or
	  (CollisionPredicates::collides_segment_point_2d(p0, p1, x1) and
	   CollisionPredicates::collides_segment_point_2d(q0, q1, x1)))
      {
#ifdef augustdebug
	std::cout << "  shewchuk swapped gave: " << tools::plot(x1) << '\n';
	for (std::size_t d = 0; d < 2; ++d)
	  std::cout << CollisionPredicates::collides_segment_point_1d(p0[d],p1[d],x1[d])<<' ';
	std::cout << '\n';
	for (std::size_t d = 0; d < 2; ++d)
	  std::cout << CollisionPredicates::collides_segment_point_1d(q0[d],q1[d],x1[d])<<' ';
	std::cout << '\n';
	std::cout << CollisionPredicates::collides_segment_point_2d(p0, p1, x1)<<' '<<CollisionPredicates::collides_segment_point_2d(q0, q1, x1)<<'\n';
#endif
	intersection.push_back(x1);
      }
      else
      {
	// Test bisection
	const bool use_p = p1.squared_distance(p0) > q1.squared_distance(q0);
	// const double alpha = numer / denom;
	// const Point& ii_intermediate = (1-alpha)*p0 + alpha*p1;
	// Point& source = use_p ?
	//   (alpha < .5 ? p0 : p1) :
	//   (ii_intermediate.squared_distance(q0) < ii_intermediate.squared_distance(q1) ? q0 : q1);
	// Point& target = use_p ?
	//   (alpha < .5 ? p1 : p0) :
	//   (ii_intermediate.squared_distance(q0) < ii_intermediate.squared_distance(q1) ? q1 : q0);

	Point& source = use_p ? p0 : q0;
	Point& target = use_p ? p1 : q1;

	Point& ref_source = use_p ? q0 : p0;
	Point& ref_target = use_p ? q1 : p1;

	// Test bisection

#ifdef augustdebug
	std::cout << "denom " << denom << " (numer = " << numer << "  alpha = "<<numer/denom<<'\n'
		  << " source " << tools::plot(source)<<'\n'
		  << " target " << tools::plot(target) << '\n'
	  // << " ii_intermediate " << tools::plot(ii_intermediate) << '\n'
	  // << " r " << tools::plot(r) << '\n'
		  << " ref_source " << tools::plot(ref_source) << '\n'
		  << " ref_target " << tools::plot(ref_target) << '\n'
		  << " orient2d ref source " << orient2d(source.coordinates(),target.coordinates(),ref_source.coordinates()) << '\n'
		  << " orient2d ref target " << orient2d(source.coordinates(),target.coordinates(),ref_target.coordinates()) << std::endl;
#ifdef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC
	std::cout<< " CGAL says: ";
	const std::vector<Point> cgal_intersection = cgal_intersection_segment_segment_2d(p0,p1,q0,q1);
	for (const Point p: cgal_intersection)
	  std::cout <<std::setprecision(std::numeric_limits<long double>::digits10+2)  << tools::plot(p,"'go','markersize',14")<<'\n';
	std::cout <<" just fyi: is the cgal inside? ";
	for (std::size_t d = 0; d < 2; ++d)
	  std::cout << CollisionPredicates::collides_segment_point_1d(p0[d],p1[d],cgal_intersection[0][d])<<' ';
	std::cout << "     ";
	for (std::size_t d = 0; d < 2; ++d)
	  std::cout << CollisionPredicates::collides_segment_point_1d(q0[d],q1[d],cgal_intersection[0][d])<<' ';
	std::cout << std::endl;
	dolfin_assert(cgal_intersection.size() == 1);
#endif

#endif

	// This should have been picked up earlier
	dolfin_assert(std::signbit(orient2d(source.coordinates(), target.coordinates(), ref_source.coordinates())) !=
		      std::signbit(orient2d(source.coordinates(), target.coordinates(), ref_target.coordinates())));



	// Preferred orientation doesn't matter in the first bisection
	const bool is_orientation_preferred = false;

	// Check solution if the lines are almost parallel
	const bool check_solution = std::abs(denom) < DOLFIN_EPS_LARGE;

	// Call bisection
	Point z0;
	const bool z0_ok = bisection(source, target, ref_source, ref_target,
				     check_solution, z0, is_orientation_preferred);
	if (z0_ok)
	{
#ifdef augustdebug
	  std::cout << "accept z0\n";
#endif
	  intersection.push_back(z0);
	}
	else
	{
	  // Call bisection again
	  //const bool check_solution = true;
	  // const bool is_orientation_preferred = false;
	  Point z1;
	  const bool z1_ok = bisection(ref_source, ref_target, source, target,
				       check_solution, z1, is_orientation_preferred);
	  if (z1_ok)
	  {
#ifdef augustdebug
	    std::cout << "accept z1\n";
#endif
	    intersection.push_back(z1);
	  }
	  else
	  {
	    // Test truncate the

	    // If z0 and z1 are close to each other, assume it's correct...
	    if (z0.squared_distance(z1) < DOLFIN_EPS)
	    {
#ifdef augustdebug
	      std::cout << "accept average\n";
#endif
	      intersection.push_back((z0 + z1) / 2);
	    }
	    else
	    {
	      // 	  // Call bisection in interval [~z0, ~z1]. However, z0 \in
	      // 	  // [source,target] and z1 \in [ref_source,ref_target]. Thus
	      // 	  // we must project either one onto the other.

	      // 	  std::cout << orient2d(ref_source.coordinates(), ref_target.coordinates(), z0.coordinates())<<' '<<orient2d(ref_source.coordinates(), ref_target.coordinates(), z1.coordinates())<<'\n'
	      // 		    << orient2d(source.coordinates(), target.coordinates(), z0.coordinates())<<' '<<orient2d(source.coordinates(), target.coordinates(), z1.coordinates())<<'\n';
	      // 	  // Get height
	      // 	  const double h0 = orient2d(source.coordinates(), target.coordinates(), z0.coordinates()) / source.distance(target);
	      // 	  const double h1 = orient2d(ref_source.coordinates(), ref_target.coordinates(), z1.coordinates()) / ref_source.distance(ref_target);
	      // 	  std::cout << "heights " << h0 <<' ' << h1 << std::endl;

	      // 	  // Project the point with the min distance
	      // 	  const bool project_z0 = std::abs(h0) < std::abs(h1) ? true : false;
	      // 	  Point a = project_z0 ? z0 : z1;
	      // 	  Point b = project_z0 ? ref_source : source;
	      // 	  Point c = project_z0 ? ref_target : target;
	      // 	  const Point r = a - c;
	      // 	  const Point s = b - c;
	      // 	  Point zproj = c + r.dot(s) / s.squared_norm() * s;
	      // 	  Point other = project_z0 ? z1 : z0;

	      // 	  // Check orientation
	      // 	  // if (std::signbit(orient2d(c.coordinates(), d.coordinates(), zproj.coordinates())) !=
	      // 	  //     std::signbit(orient2d(c.coordinates(), d.coordinates(), other.coordinates())))
	      // 	  std::cout << orient2d(b.coordinates(), c.coordinates(), zproj.coordinates())<<' '<< orient2d(b.coordinates(), c.coordinates(), other.coordinates())<<std::endl;

	      // 	  Point z01;
	      // 	  bool z01_ok;

	      // 	  if (std::signbit(orient2d(b.coordinates(), c.coordinates(), zproj.coordinates())) != std::signbit(orient2d(b.coordinates(), c.coordinates(), other.coordinates())))
	      // 	  {
	      // #ifdef augustdebug
	      // 	    std::cout << "bisect using project_z0 " << project_z0 << '\n';
	      // #endif
	      // 	    // Call bisection
	      // 	    z01_ok = bisection(zproj, other, b, c,
	      // 			       check_solution, z01, is_orientation_preferred);
	      // 	  }
	      // 	  else
	      // 	  {
	      // #ifdef augustdebug
	      // 	    std::cout << "bisect using other point\n";
	      // #endif
	      // 	    // Take the other point
	      // 	    Point a = !project_z0 ? z0 : z1;
	      // 	    Point b = !project_z0 ? ref_source : source;
	      // 	    Point c = !project_z0 ? ref_target : target;
	      // 	    const Point r = a - c;
	      // 	    const Point s = b - c;
	      // 	    Point zproj = c + r.dot(s) / s.squared_norm() * s;
	      // 	    Point other = project_z0 ? z1 : z0;
	      // 	    std::cout << orient2d(b.coordinates(), c.coordinates(), zproj.coordinates())<<' '<< orient2d(b.coordinates(), c.coordinates(), other.coordinates())<<std::endl;
	      // 	    // Call bisection
	      // 	    z01_ok = bisection(zproj, other, b, c,
	      // 			       check_solution, z01, is_orientation_preferred);
	      // 	  }

	      // 	  if (z01_ok)
	      // 	  {
	      // #ifdef augustdebug
	      // 	    std::cout << "accept z01\n";
	      // #endif
	      // 	    intersection.push_back(z01);
	      // 	  }
	      // 	  else
	      // 	  {
	      // 	    PPause;
	      // 	  }


	      // Call bisection in interval [z0, X], [z1, Y]. z0 \in
	      // [source,target].
	      const double z0_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), z0.coordinates());
	      const double s_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), source.coordinates());
	      Point& X = (z0_orientation*s_orientation >= 0) ? target : source;
	      const double z1_orientation = orient2d(source.coordinates(), target.coordinates(), z1.coordinates());
	      const double rs_orientation = orient2d(source.coordinates(), target.coordinates(), ref_source.coordinates());
	      Point& Y = (z1_orientation*rs_orientation >= 0) ? ref_target : ref_source;

#ifdef augustdebug
	      std::cout << "new interval for bisection with orientation\n";
	      const double t_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), target.coordinates());
	      std::cout << "z0 s t orientations: " << z0_orientation<<' '<<s_orientation<<' '<<t_orientation << '\n';
	      const double rt_orientation = orient2d(source.coordinates(), target.coordinates(), ref_target.coordinates());
	      std::cout << "z1 rs rt orientations: " << z1_orientation <<' '<<rs_orientation<<' '<<rt_orientation << '\n';
	      std::cout<<"chose X = " << ((X == target) ? "target\n" : " source\n")
		       <<"chose Y = " << ((Y == ref_target) ? "ref_target\n" : "ref_source\n");
	      std::cout <<"distances z0 X " << z0.distance(X)<<" z1 Y " << z1.distance(Y)<<'\n';
	      std::cout << "z1-Y vs z0 and X: "<<orient2d(z1.coordinates(), Y.coordinates(), z0.coordinates())<<' '
			<<orient2d(z1.coordinates(), Y.coordinates(), X.coordinates()) << std::endl;
	      std::cout << " test z1 Y vs source, target, z0: " << orient2d(z1.coordinates(),  Y.coordinates(), source.coordinates())<<' '<<orient2d(z1.coordinates(), Y.coordinates(), target.coordinates())<<' ' << orient2d(z1.coordinates(), Y.coordinates(), z0.coordinates())<<'\n';
	      std::cout << "s t " << tools::plot(source)<<tools::plot(target)<<'\n'
			<< "rs rt " << tools::plot(ref_source)<<tools::plot(ref_target) << '\n';
#endif


	      Point z01;
	      const bool z01_ok = bisection(z0, X, z1, Y,
					    check_solution, z01, is_orientation_preferred);
	      if (z01_ok)
	      {
#ifdef augustdebug
		std::cout << "accept z01\n";
#endif
		intersection.push_back(z01);
	      }
	      else
	      {
		PPause;
// #ifdef augustdebug
// 		std::cout << "test perturb largest point and dim\n";
// #endif
// 		std::array<Point, 4> pts = {p0, p1, q0, q1};
// 		double maxp = p0[0];
// 		std::size_t point_number = 0; // use 0,1,2,3,4,5,6,7 for p0[0], p0[1], ..., q1[1]
// 		for (std::size_t i = 0; i < 4; ++i)
// 		  for (std::size_t d = 0; d < 2; ++d)
// 		    if (std::abs(pts[i][d]) > maxp)
// 		    {
// 		      maxp = std::abs(pts[i][d]);
// 		      point_number = 2*i + d;
// 		    }
// #ifdef augustdebug
// 		std::cout << "point no " << point_number/2<<" dim " << point_number%2 << std::endl;
// #endif
// 		pts[point_number/2][point_number%2] += std::numeric_limits<double>::epsilon();
// 		std::vector<Point> intersection_perturbed = _intersection_segment_segment_2d(pts[0], pts[1], pts[2], pts[3]);

// 		for (const Point p: intersection_perturbed)
// 		  std::cout << p<<std::endl;
// 		PPause;
	      }


	      // 	  // z0 and z1 have different orientation
	      // 	  Point z01;
	      // 	  bool z01_ok;
	      // 	  if (std::signbit(orient2d(source.coordinates(), target.coordinates(), z0.coordinates())) !=
	      // 	      std::signbit(orient2d(source.coordinates(), target.coordinates(), z1.coordinates())))
	      // 	  {
	      // #ifdef augustdebug
	      // 	    std::cout << "try with source and target\n";
	      // #endif
	      // 	    z01_ok = bisection(z0, z1, source, target,
	      // 			       check_solution, z01, is_orientation_preferred);
	      // 	  }
	      // 	  else if (std::signbit(orient2d(ref_source.coordinates(), ref_target.coordinates(), z0.coordinates())) !=
	      // 		   std::signbit(orient2d(ref_source.coordinates(), ref_target.coordinates(), z1.coordinates())))
	      // 	  {
	      // #ifdef augustdebug
	      // 	    std::cout << "try with ref_source and ref_target\n";
	      // #endif
	      // 	    z01_ok = bisection(z0, z1, ref_source, ref_target,
	      // 			       check_solution, z01, is_orientation_preferred);
	      // 	  }
	      // 	  else
	      // 	  {
	      // 	    PPause;
	      // 	  }

	      // 	  if (z01_ok)
	      // 	  {
	      // #ifdef augustdebug
	      // 	    std::cout << "accept z01\n";
	      // #endif
	      // 	    intersection.push_back(z01);
	      // 	  }
	      // 	  else
	      // 	  {
	      // 	    PPause;
	      // 	  }

	    }
	  }
	}
      } // end if (denom != 0)
      // else // Not parallel and no intersection
      // {
      //   // std::cout << "case 5"<<std::endl;
      //   return intersection;
      // }
    }
  }

  //     Point z;
  //     bisection(source, target, ref_source, ref_target, false, z, false);

  //     // Check the solution z if the lines are almost parallel
  //     if (std::abs(denom) < DOLFIN_EPS_LARGE)
  //     {
  //       // We can check z in a number of ways:

  //       // Is it inside the two domains?  (tempting to assert
  //       // collides_segment_point_1d or 2d, but this is not possible: if
  //       // we have a horizontal line it is difficult to assert a point
  //       // exactly in this direction. Collides segment point 2d is also
  //       // difficult)

  //       // check that z is inside the smallest interval
  //       std::vector<std::pair<double,Point> > dists;
  //       dists.emplace_back(z.squared_distance(p0), p0);
  //       dists.emplace_back(z.squared_distance(p1), p1);
  //       dists.emplace_back(z.squared_distance(q0), q0);
  //       dists.emplace_back(z.squared_distance(q1), q1);
  //       std::sort(dists.begin(), dists.end(), [](std::pair<double,Point> left,
  // 					       std::pair<double,Point> right) {
  // 		  return left.first < right.first;
  // 		});
  //       const Point r = (use_p) ? p1 - p0 : q1 - q0;
  //       const double t0 = r.dot(z-dists[0].second);
  //       const double t1 = r.dot(z-dists[1].second);
  //       const double t2 = r.dot(z-dists[2].second);
  //       const double t3 = r.dot(z-dists[3].second);
  //       int cnt=0;
  //       if (t0<0) cnt++;
  //       if (t1<0) cnt++;
  //       if (t2<0) cnt++;
  //       if (t3<0) cnt++;
  // #ifdef augustdebug
  //       std::cout << t0<<' '<<t1<<' '<<t2<<' '<<t3<<std::endl;
  // #endif
  //       const bool all_inside =
  // 	CollisionPredicates::collides_segment_point_1d(p0[0],p1[0],z[0]) and
  // 	CollisionPredicates::collides_segment_point_1d(p0[1],p1[1],z[1]) and
  // 	CollisionPredicates::collides_segment_point_1d(q0[0],q1[0],z[0]) and
  // 	CollisionPredicates::collides_segment_point_1d(q0[1],q1[1],z[1]);

  //       if (cnt!=2) {
  // 	if (std::abs(t0)>DOLFIN_EPS or std::abs(t1)>DOLFIN_EPS or
  // 	    std::abs(t2)>DOLFIN_EPS or std::abs(t3)>DOLFIN_EPS or
  // 	    !all_inside)
  // 	{
  // 	  //std::cout << tools::generate_test(a,b,c,d,__FUNCTION__)<<std::endl;
  // 	  //PPause;

  // #ifdef augustdebug
  // 	  std::cout << "call bisection again with points swapped\n";
  // #endif
  // 	  Point z1;
  // 	  bisection(ref_source, ref_target, source, target, false, z1, false);

  // 	  std::vector<std::pair<double,Point> > dists1;
  // 	  dists1.emplace_back(z1.squared_distance(p0), p0);
  // 	  dists1.emplace_back(z1.squared_distance(p1), p1);
  // 	  dists1.emplace_back(z1.squared_distance(q0), q0);
  // 	  dists1.emplace_back(z1.squared_distance(q1), q1);
  // 	  std::sort(dists.begin(), dists.end(), [](std::pair<double,Point> left,
  // 						   std::pair<double,Point> right) {
  // 		      return left.first < right.first;
  // 		    });
  // 	  const Point r = (use_p) ? p1 - p0 : q1 - q0;
  // 	  const double t0 = r.dot(z1-dists1[0].second);
  // 	  const double t1 = r.dot(z1-dists1[1].second);
  // 	  const double t2 = r.dot(z1-dists1[2].second);
  // 	  const double t3 = r.dot(z1-dists1[3].second);
  // 	  int cnt=0;
  // 	  if (t0<0) cnt++;
  // 	  if (t1<0) cnt++;
  // 	  if (t2<0) cnt++;
  // 	  if (t3<0) cnt++;

  // #ifdef augustdebug
  // 	  // test collision
  // 	  for (std::size_t d = 0; d < 2; ++d)
  // 	    std::cout << CollisionPredicates::collides_segment_point_1d(p0[d],p1[d],z1[d])<<' ';
  // 	  std::cout << std::endl;
  // 	  for (std::size_t d = 0; d < 2; ++d)
  // 	    std::cout << CollisionPredicates::collides_segment_point_1d(q0[d],q1[d],z1[d])<<' ';
  // 	  std::cout << std::endl;
  // 	  std::cout << t0<<' '<<t1<<' '<<t2<<' '<<t3<<std::endl;
  // #endif

  // 	  const bool all_inside =
  // 	    CollisionPredicates::collides_segment_point_1d(p0[0],p1[0],z1[0]) and
  // 	    CollisionPredicates::collides_segment_point_1d(p0[1],p1[1],z1[1]) and
  // 	    CollisionPredicates::collides_segment_point_1d(q0[0],q1[0],z1[0]) and
  // 	    CollisionPredicates::collides_segment_point_1d(q0[1],q1[1],z1[1]);

  // 	  //dolfin_assert(cnt==2);
  // 	  if (cnt!=2) {
  // 	    if (std::abs(t0)>DOLFIN_EPS or std::abs(t1)>DOLFIN_EPS or
  // 		std::abs(t2)>DOLFIN_EPS or std::abs(t3)>DOLFIN_EPS or
  // 		!all_inside)
  // 	    {
  // #ifdef augustdebug
  // 	      std::cout << "possible error, take closest which is of distance = " << dists[0].first <<" or " << dists1[0].first << " (points are " << dists[0].second << " and " << dists1[0].second << std::endl;
  // #endif

  // 	      //dolfin_assert(dists[0].first < DOLFIN_EPS or dists1[0].first < DOLFIN_EPS);
  // 	      if (dists[0].first < DOLFIN_EPS or dists1[0].first < DOLFIN_EPS)
  // 	      {
  // 		z = dists[0].first < dists1[0].first ? z : z1;
  // 	      }
  // 	      else
  // 	      {
  // 		dists.clear(); dists1.clear();

  // 		// z = z1;
  // 		// std::cout << "possible error, check what CGAL says\n";
  // 		// const std::vector<Point> cgal_intersection = cgal_intersection_segment_segment_2d(a,b,c,d);
  // 		// for (const Point p: cgal_intersection)
  // 		// 	std::cout << p << std::endl;

  // #ifdef augustdebug
  // 		std::cout << "do bisection with the points found as start points\n";
  // #endif

  // 		// Redo previous bisection for z1 and pick a preferred
  // 		// orientation sign of the point obtained. This is
  // 		// needed if we need to do the bisection a third time,
  // 		// using z and z1 as target and source, because they
  // 		// need to have different orientation. This shouldn't
  // 		// matter for the actual point obtained (the
  // 		// difference should be exactly DOLFIN_EPS in the
  // 		// error b-a, meaning that the difference for the
  // 		// point should be of the order
  // 		// DOLFIN_EPS*std::max(point.coordinates())).

  // 		// Should we do z,z1 or z1,z?
  // 		double z_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), z.coordinates());
  // 		double z1_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), z1.coordinates());
  // 		Point z2;
  // 		if (std::signbit(z_orientation) != std::signbit(z1_orientation))
  // 		{
  // 		  bisection(z,z1,ref_source,ref_target,false, z2, false);
  // 		}
  // 		else
  // 		{
  // 		  // Recompute either z or z1
  // 		  const bool preferred_orientation_sign = !std::signbit(orient2d(ref_source.coordinates(), ref_target.coordinates(), z.coordinates()));
  // 		  bisection(ref_source, ref_target, source, target, false,z1,false);
  // 			    //true, preferred_orientation_sign);
  // 		  double z_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), z.coordinates());
  // 		  double z1_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), z1.coordinates());
  // #ifdef augustdebug
  // 		  std::cout << z_orientation<<' '<<z1_orientation<<std::endl;
  // #endif
  // 		  PPause;
  // 		}

  // 		std::vector<std::pair<double,Point> > dists2;
  // 		dists2.emplace_back(z2.squared_distance(p0), p0);
  // 		dists2.emplace_back(z2.squared_distance(p1), p1);
  // 		dists2.emplace_back(z2.squared_distance(q0), q0);
  // 		dists2.emplace_back(z2.squared_distance(q1), q1);
  // 		std::sort(dists2.begin(), dists2.end(), [](std::pair<double,Point> left,
  // 							   std::pair<double,Point> right) {
  // 			    return left.first < right.first;
  // 			  });
  // 		const Point r = (use_p) ? p1 - p0 : q1 - q0;
  // 		const double t0 = r.dot(z2-dists2[0].second);
  // 		const double t1 = r.dot(z2-dists2[1].second);
  // 		const double t2 = r.dot(z2-dists2[2].second);
  // 		const double t3 = r.dot(z2-dists2[3].second);
  // 		int cnt=0;
  // 		if (t0<0) cnt++;
  // 		if (t1<0) cnt++;
  // 		if (t2<0) cnt++;
  // 		if (t3<0) cnt++;

  // #ifdef augustdebug
  // 		// test collision
  // 		for (std::size_t d = 0; d < 2; ++d)
  // 		  std::cout << CollisionPredicates::collides_segment_point_1d(p0[d],p1[d],z2[d])<<' ';
  // 		std::cout << std::endl;
  // 		for (std::size_t d = 0; d < 2; ++d)
  // 		  std::cout << CollisionPredicates::collides_segment_point_1d(q0[d],q1[d],z2[d])<<' ';
  // 		std::cout << std::endl;
  // 		std::cout << t0<<' '<<t1<<' '<<t2<<' '<<t3<<std::endl;
  // #endif
  // 		const bool all_inside =
  // 		  CollisionPredicates::collides_segment_point_1d(p0[0],p1[0],z2[0]) and
  // 		  CollisionPredicates::collides_segment_point_1d(p0[1],p1[1],z2[1]) and
  // 		  CollisionPredicates::collides_segment_point_1d(q0[0],q1[0],z2[0]) and
  // 		  CollisionPredicates::collides_segment_point_1d(q0[1],q1[1],z2[1]);

  // 		if (cnt!=2) {
  // 		  if (std::abs(t0)>DOLFIN_EPS or std::abs(t1)>DOLFIN_EPS or
  // 		      std::abs(t2)>DOLFIN_EPS or std::abs(t3)>DOLFIN_EPS or
  // 		      !all_inside)
  // 		  {

  // #ifdef augustdebug
  // 		    std::cout << "possible error, dists2 min = " << dists2[0].first << std::endl;
  // #endif
  // 		    if (dists2[0].first < DOLFIN_EPS)
  // 		    {
  // 		      z = z2;
  // 		    }
  // 		    else if (std::abs(t0-t1) < DOLFIN_EPS_LARGE)
  // 		    {
  // 		      // meaning that these the two closest points found to z are close. Take mean of these
  // 		      const Point z3 = (dists2[0].second + dists2[1].second) / 2;
  // 		      const bool all_inside =
  // 			CollisionPredicates::collides_segment_point_1d(p0[0],p1[0],z3[0]) and
  // 		    	CollisionPredicates::collides_segment_point_1d(p0[1],p1[1],z3[1]) and
  // 		    	CollisionPredicates::collides_segment_point_1d(q0[0],q1[0],z3[0]) and
  // 		    	CollisionPredicates::collides_segment_point_1d(q0[1],q1[1],z3[1]);
  // 		      dolfin_assert(all_inside);
  // #ifdef augustdebug
  // 		      std::cout << "take average\n";
  // #endif
  // 		      z = z3;
  // 		    }
  // 		    else
  // 		    {
  // 		      PPause;
  // 		    }
  // 		  }
  // 		}
  // 		else
  // 		{
  // #ifdef augustdebug
  // 		  std::cout << "new point z2 accepted\n";
  // #endif
  // 		  z = z2;
  // 		}
  // 	      }

  // 	    }
  // 	  }
  // 	  else
  // 	  {
  // #ifdef augustdebug
  // 	    std::cout << "new point z1 accepted\n";
  // #endif

  // 	    z = z1;
  // 	  }

  // 	}
  //       }
  //     }

  //     intersection.push_back(z);
  //   } // end if (denom != 0)
  // else // Not parallel and no intersection
  // {
  //   // std::cout << "case 5"<<std::endl;
  //   return intersection;
  // }


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

  // std::cout << "case 6"<<std::endl;
  // return intersection;

  // std::cout << __FUNCTION__<< "points " << tools::plot(p0)<<tools::plot(p1)<<tools::plot(q0)<<tools::plot(q1)<<std::endl;


  // std::vector<Point> intersection;

  // // Add vertex-vertex collision to the intersection
  // if (p0 == q0 or p0 == q1)
  //   intersection.push_back(p0);
  // if (p1 == q0 or p1 == q1)
  //   intersection.push_back(p1);

  // // Add vertex-"segment interior" collisions to the intersection
  // if (CollisionPredicates::collides_interior_point_segment_2d(q0, q1, p0))
  //   intersection.push_back(p0);

  // if (CollisionPredicates::collides_interior_point_segment_2d(q0, q1, p1))
  //   intersection.push_back(p1);

  // if (CollisionPredicates::collides_interior_point_segment_2d(p0, p1, q0))
  //   intersection.push_back(q0);

  // if (CollisionPredicates::collides_interior_point_segment_2d(p0, p1, q1))
  //   intersection.push_back(q1);

  // if (intersection.empty())
  // {
  //   std::cout << ' '<<__FUNCTION__<<" gave empty intersection: must check interior\n";

  //   // No collisions in any vertices, so check interior
  //   return _intersection_segment_interior_segment_interior_2d(p0, p1, q0, q1);
  // }

  // return intersection;

}
//-----------------------------------------------------------------------------
// Note that for parallel segments, only vertex-"edge interior" collisions will
// be returned
std::vector<Point>
IntersectionConstruction::_intersection_segment_interior_segment_interior_2d(Point p0,
                                                                             Point p1,
                                                                             Point q0,
                                                                             Point q1)
{
  // Shewchuk style
  const double q0_q1_p0 = orient2d(q0.coordinates(),
                                   q1.coordinates(),
                                   p0.coordinates());
  const double q0_q1_p1 = orient2d(q0.coordinates(),
                                   q1.coordinates(),
                                   p1.coordinates());
  const double p0_p1_q0 = orient2d(p0.coordinates(),
                                   p1.coordinates(),
                                   q0.coordinates());
  const double p0_p1_q1 = orient2d(p0.coordinates(),
                                   p1.coordinates(),
                                   q1.coordinates());

  std::cout << __FUNCTION__<< " points: " << tools::plot(p0)<<tools::plot(p1)<<tools::plot(q0)<<tools::plot(q1)<<'\n'
	    <<' '<<q0_q1_p0<<' '<<q0_q1_p1<<' '<<p0_p1_q0<<' '<<p0_p1_q1<<std::endl;

  std::cout << " test collision: " << CollisionPredicates::collides_interior_point_segment_2d(p0,p1,q0)<<' '<<CollisionPredicates::collides_interior_point_segment_2d(p0,p1,q1)<<std::endl;

  std::vector<Point> intersection;

  if (q0_q1_p0 != 0 && q0_q1_p1 != 0 && p0_p1_q0 != 0 && p0_p1_q1 != 0 &&
      std::signbit(q0_q1_p0) != std::signbit(q0_q1_p1) && std::signbit(p0_p1_q0) != std::signbit(p0_p1_q1))
  {
    // Segments intersect in both's interior.
    // Compute intersection
    const double denom = (p1.x()-p0.x())*(q1.y()-q0.y()) - (p1.y()-p0.y())*(q1.x()-q0.x());
    const double numerator = q0_q1_p0;
    const double alpha = numerator/denom;

    if (std::abs(denom) < DOLFIN_EPS_LARGE)
    {
      std::cout << __FUNCTION__ << "points " << p0<<' '<<p1<<' '<<q0<<' '<<q1<<'\n'
      		<< " parallel numerator/denomenator="<<numerator<<" / " << denom << " = " << alpha << std::endl;
      std::cout << tools::drawtriangle({p0,p1})<<tools::drawtriangle({q0,q1})<<std::endl;


      // Test exact arithmetic for the denominator
      {
	typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(128)> TT;
	const TT p0x(p0.x()), p0y(p0.y()), p1x(p1.x()), p1y(p1.y()), q0x(q0.x()), q0y(q0.y()), q1x(q1.x()), q1y(q1.y()), numerator_tt(numerator);
	const TT denom_tt = (p1.x()-p0.x())*(q1.y()-q0.y()) - (p1.y()-p0.y())*(q1.x()-q0.x());

	std::cout <<"denom zero? " << (denom_tt == 0) << " p0==p1 " << (p0==p1)<<" q0 == q1 " << (q0==q1)<<" dists " << (p1-p0).squared_norm() << ' ' << (q1-q0).squared_norm() << std::endl;

	if (denom_tt == 0) // exactly parallel
	{
	  // Take the longest distance as a,b
	  Point a = p0, b = p1, c = q0, d = q1;
	  if (a.squared_distance(b) < c.squared_distance(d))
	  {
	    std::swap(a, c);
	    std::swap(b, d);
	  }

	  // Assume line is l(t) = a + t*v, where v = b-a. Find location of c and d:
	  const Point v = b - a;
	  const double vnorm2 = v.squared_norm();

	  // FIXME Investigate this further if vnorm2 is small
	  dolfin_assert(vnorm2 > DOLFIN_EPS);

	  const double tc = v.dot(c - a) / vnorm2;
	  const double td = v.dot(d - a) / vnorm2;
	  // Find if c and d are to the left or to the right. Remember
	  // that we assume there is a collision between ab and cd.
	  bool found_a = false;
	  bool found_b = false;
	  if (tc > 0)
	  {
	    if (tc < 1) // tc is now 0 < tc < 1 => between a and b
	      intersection.push_back(c);
	    else // tc must be larger than b
	    {
	      intersection.push_back(b);
	      found_b = true;
	    }
	  }
	  else // tc is to the left of a
	  {
	    intersection.push_back(a);
	    found_a = true;
	  }

	  if (td > 0)
	  {
	    if (td < 1)
	      intersection.push_back(d);
	    else
	    {
	      dolfin_assert(!found_b);
	      intersection.push_back(b);
	    }
	  }
	  else
	  {
	    dolfin_assert(!found_a);
	    intersection.push_back(a);
	  }
	  std::cout << "% intersection(s):\n";
	  for (const Point p: intersection)
	    std::cout << tools::plot(p,"'gx'");
	  std::cout << std::endl;


	}
	else
	{

	  const TT alpha_tt = numerator_tt / denom_tt;

	  // std::cout << "numerator_tt / denom_tt = alpha_tt = " << numerator_tt << ' ' << denom_tt << ' ' << alpha_tt << std::endl;


	  const TT n_tt = orient2d(q0.coordinates(), q1.coordinates(), p1.coordinates());


	  intersection.push_back(alpha > .5 ?
				 //p1 - n / denom * (p0 - p1) :
				 p1 * (1 + (n_tt / denom_tt).ToDouble()) - p0 * (n_tt / denom_tt).ToDouble() :
				 //p0 + numerator / denom * (p1 - p0)
				 p0 * (1 - alpha_tt.ToDouble()) + p1 * alpha_tt.ToDouble()
				 );

	  // std::cout << "% intersection(s):\n";
	  // for (const Point p: intersection)
	  //   std::cout << tools::plot(p,"'gx'");
	  // std::cout << std::endl;


	  // ttmath::UInt<2> a,b,c;

	  // a = "1234";
	  // b = 3456;
	  // c = a*b;

	  // std::cout << c << " exit "<<std::endl;
	  // exit(0);
	}
      } // end test of exact arithmetic



      // // FIXME: assume we have parallel lines that intersect. This needs further testing
      // Point a = p0, b = p1, c = q0, d = q1;

      // // Take the longest distance as a,b
      // if (a.squared_distance(b) < c.squared_distance(d))
      // {
      // 	std::swap(a, c);
      // 	std::swap(b, d);
      // }

      // // Assume line is l(t) = a + t*v, where v = b-a. Find location of c and d:
      // const Point v = b - a;
      // const double vnorm2 = v.squared_norm();

      // // FIXME Investigate this further if vnorm2 is small
      // dolfin_assert(vnorm2 > DOLFIN_EPS);

      // const double tc = v.dot(c - a) / vnorm2;
      // const double td = v.dot(d - a) / vnorm2;

      // // Find if c and d are to the left or to the right. Remember
      // // that we assume there is a collision between ab and cd.
      // bool found_a = false;
      // bool found_b = false;
      // if (tc > 0)
      // {
      // 	if (tc < 1) // tc is now 0 < tc < 1 => between a and b
      // 	  intersection.push_back(c);
      // 	else // tc must be larger than b
      // 	{
      // 	  intersection.push_back(b);
      // 	  found_b = true;
      // 	}
      // }
      // else // tc is to the left of a
      // {
      // 	intersection.push_back(a);
      // 	found_a = true;
      // }

      // if (td > 0)
      // {
      // 	if (td < 1)
      // 	  intersection.push_back(d);
      // 	else
      // 	{
      // 	  dolfin_assert(!found_b);
      // 	  intersection.push_back(b);
      // 	}
      // }
      // else
      // {
      // 	dolfin_assert(!found_a);
      // 	intersection.push_back(a);
      // }

      // std::cout << "% intersection(s):\n";
      // for (const Point p: intersection)
      // 	std::cout << tools::plot(p,"'gx'");
      // std::cout << std::endl;




      // // Segment are almost parallel, so result may vulnerable to roundoff
      // // errors.
      // // Let's do an iterative bisection instead

      // // FIXME: Investigate using long double for even better precision
      // // or fall back to exact arithmetic?

      // const bool use_p = p1.squared_distance(p0) > q1.squared_distance(q0);
      // const Point& ii_intermediate = p0 + alpha*(p1-p0);
      // Point& source = use_p ? (alpha < .5 ? p0 : p1) : (ii_intermediate.squared_distance(q0) < ii_intermediate.squared_distance(q1) ? q0 : q1);
      // Point& target = use_p ? (alpha < .5 ? p1 : p0) : (ii_intermediate.squared_distance(q0) < ii_intermediate.squared_distance(q1) ? q1 : q0);

      // Point& ref_source = use_p ? q0 : p0;
      // Point& ref_target = use_p ? q1 : p1;

      // dolfin_assert(std::signbit(orient2d(source.coordinates(), target.coordinates(), ref_source.coordinates())) !=
      //               std::signbit(orient2d(source.coordinates(), target.coordinates(), ref_target.coordinates())));

      // // Shewchuk notation
      // dolfin::Point r = target-source;

      // int iterations = 0;
      // double a = 0;
      // double b = 1;

      // const double source_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), source.coordinates());
      // double a_orientation = source_orientation;
      // double b_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), target.coordinates());

      // while (std::abs(b-a) > DOLFIN_EPS_LARGE)
      // {
      //   dolfin_assert(std::signbit(orient2d(ref_source.coordinates(), ref_target.coordinates(), (source+a*r).coordinates())) !=
      //                 std::signbit(orient2d(ref_source.coordinates(), ref_target.coordinates(), (source+b*r).coordinates())));

      //   const double new_alpha = (a+b)/2;
      //   dolfin::Point new_point = source+new_alpha*r;
      //   const double mid_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), new_point.coordinates());

      //   if (mid_orientation == 0)
      //   {
      //     a = new_alpha;
      //     b = new_alpha;
      //     break;
      //   }

      //   if (std::signbit(source_orientation) == std::signbit(mid_orientation))
      //   {
      //     a_orientation = mid_orientation;
      //     a = new_alpha;
      //   }
      //   else
      //   {
      //     b_orientation = mid_orientation;
      //     b = new_alpha;
      //   }

      //   iterations++;
      // }

      // if (a == b)
      //   intersection.push_back(source + a*r);
      // else
      //   intersection.push_back(source + (a+b)/2*r);
    }
    else
    {
      std::cout << "denom is not small, but numerator/denom = alpha = " << numerator<<" / " << denom << " = " << alpha << std::endl;

      typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(128)> TT;
      const TT p0x(p0.x()), p0y(p0.y()), p1x(p1.x()), p1y(p1.y()), q0x(q0.x()), q0y(q0.y()), q1x(q1.x()), q1y(q1.y()), numerator_tt(numerator);
      const TT denom_tt = (p1.x()-p0.x())*(q1.y()-q0.y()) - (p1.y()-p0.y())*(q1.x()-q0.x());
      const TT alpha_tt = numerator_tt / denom_tt;

      std::cout << " or with TT: numerator_tt / denom_tt = alpha_tt = " << numerator_tt << ' ' << denom_tt << ' ' << alpha_tt << std::endl;

      intersection.push_back(alpha > .5 ? p1 - orient2d(q0.coordinates(), q1.coordinates(), p1.coordinates())/denom * (p0-p1) : p0 + numerator/denom * (p1-p0));
    }
  }

  return intersection;
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_segment_segment_3d(const Point& p0,
							   const Point& p1,
							   const Point& q0,
							   const Point& q1)
{
  dolfin_error("IntersectionConstruction.cpp",
	       "compute segment-segment 3d intersection",
	       "Not implemented.");
  return std::vector<Point>();
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
IntersectionConstruction::_intersection_triangle_segment_3d(const Point& p0,
							    const Point& p1,
							    const Point& p2,
							    const Point& q0,
							    const Point& q1)
{
  dolfin_error("IntersectionConstruction.cpp",
	       "compute triangle-segment 3d intersection",
	       "Not implemented.");
  return std::vector<Point>();
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

  // std::vector<Point> points_0 = intersection_triangle_segment_2d(p0, p1, p2,
  // 								 q0, q1);
  // std::vector<Point> points_1 = intersection_triangle_segment_2d(p0, p1, p2,
  // 								 q0, q2);
  // std::vector<Point> points_2 = intersection_triangle_segment_2d(p0, p1, p2,
  // 								 q1, q2);
  // std::vector<Point> points_3 = intersection_triangle_segment_2d(q0, q1, q2,
  // 								 p0, p1);
  // std::vector<Point> points_4 = intersection_triangle_segment_2d(q0, q1, q2,
  // 								 p0, p2);
  // std::vector<Point> points_5 = intersection_triangle_segment_2d(q0, q1, q2,
  // 								 p1, p2);
  // std::vector<Point> points;
  // points.insert(points.end(),
  // 		points_0.begin(), points_0.end());
  // points.insert(points.end(),
  // 		points_1.begin(), points_1.end());
  // points.insert(points.end(),
  // 		points_2.begin(), points_2.end());
  // points.insert(points.end(),
  // 		points_3.begin(), points_3.end());
  // points.insert(points.end(),
  // 		points_4.begin(), points_4.end());
  // points.insert(points.end(),
  // 		points_5.begin(), points_5.end());

  // // Remove strict duplictes. Use exact equality here. Approximate
  // // equality is for ConvexTriangulation.
  // // FIXME: This can be avoided if we use interior segment tests.
  // const std::vector<Point> unique = unique_points(points);

  // // std::cout << __FUNCTION__<<" intersection of\n"
  // // 	    << tools::drawtriangle({p0,p1,p2})<<tools::drawtriangle({q0,q1,q2})<<std::endl<<" gave these points (if any): ";
  // // for (const Point p: unique_points)
  // //   std::cout << tools::plot(p);
  // // std::cout << std::endl;

  // return unique;

  // std::cout << __FUNCTION__<<" "<<tools::drawtriangle({p0,p1,p2})<<tools::drawtriangle({q0,q1,q2})<<std::endl;

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
    for (const Point p: points) std::cout << tools::plot(p);
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
    for (const Point p: points) std::cout << tools::plot(p);
    std::cout << std::endl;
#endif

    // Find all "edge interior"-"edge interior" intersections
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = 0; j < 3; j++)
      {
  	{
  	  // std::vector<Point> triangulation =
  	  //   intersection_segment_interior_segment_interior_2d(tri_0[i],
  	  // 						      tri_0[(i+1)%3],
  	  // 						      tri_1[j],
  	  // 						      tri_1[(j+1)%3]);
  	  std::vector<Point> triangulation =
  	    intersection_segment_segment_2d(tri_0[i],
					    tri_0[(i+1)%3],
					    tri_1[j],
					    tri_1[(j+1)%3]);
	  // // FIXME Remove edge vertices
	  // std::vector<Point> triangulation;
	  // for (const Point& p: triangulation_tmp)
	  // {
	  //   if (p != tri_0[i] and p != tri_0[(i+1)%3] and
	  // 	p != tri_1[i] and p != tri_1[(i+1)%3])
	  //     triangulation.push_back(p);
	  // }

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
  std::array<Point, 4> tet_0 = {p0, p1, p2, p3};
  std::array<Point, 4> tet_1 = {q0, q1, q2, q3};

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

  // Remove duplicate nodes
  // FIXME: If this is necessary, reuse code from ConvexTriangulation
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


}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::unique_points(std::vector<Point> input_points)
{
  // Create a strictly unique list of points
  std::sort(input_points.begin(), input_points.end(), operator<);
  const std::vector<Point>::iterator last = std::unique(input_points.begin(), input_points.end(), operator==);
  input_points.erase(last, input_points.end());
  return input_points;
}

//-----------------------------------------------------------------------------
bool IntersectionConstruction::bisection(Point source,
					 Point target,
					 Point ref_source,
					 Point ref_target,
					 bool check_solution,
					 Point& z,
					 bool is_orientation_set,
					 bool orientation)
{

  int iterations = 0;
  double a = 0;
  double b = 1;

  const double source_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), source.coordinates());
  double a_orientation = source_orientation;
  double b_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), target.coordinates());

#ifdef augustdebug
  std::cout << "a_orientation " << a_orientation << '\n'
	    << "b_orientation " << b_orientation << std::endl;
#endif

  // initalize to keep compiler happy
  double new_alpha = 0;
  Point new_point = source;
  double mid_orientation = 0;

  while (std::abs(b-a) > DOLFIN_EPS)
  {
    // #ifdef augustdebug
    // 	std::cout << orient2d(ref_source.coordinates(), ref_target.coordinates(), ((1-a)*source+a*target).coordinates()) <<' '<<orient2d(ref_source.coordinates(), ref_target.coordinates(), ((1-b)*source+b*target).coordinates())<<' '<< tools::plot((1-b)*source+b*target)<<' '<<tools::plot(target) <<orient2d(ref_source.coordinates(), ref_target.coordinates(), target.coordinates())<< ' '<<orient2d(ref_source.coordinates(),ref_target.coordinates(), ((1-b)*source+b*target).coordinates())<<std::endl;
    // #endif
    dolfin_assert
      (std::signbit(orient2d(ref_source.coordinates(),
			     ref_target.coordinates(),
			     (a == 0) ? source.coordinates() : ((1-a)*source+a*target).coordinates())) !=
       std::signbit(orient2d(ref_source.coordinates(),
			     ref_target.coordinates(),
			     (a == 0) ? target.coordinates() : ((1-b)*source+b*target).coordinates())));

    new_alpha = (a+b)/2;
    // new_point = (1-new_alpha)*source+new_alpha*target;
    new_point = ((2-a-b)*source+(a+b)*target) / 2;

    mid_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), new_point.coordinates());

    // test
    if (mid_orientation == 0)
    {
      a = new_alpha;
      b = new_alpha;
      break;
    }

    if (std::signbit(source_orientation) == std::signbit(mid_orientation))
    {
      a_orientation = mid_orientation;
      a = new_alpha;
    }
    else
    {
      b_orientation = mid_orientation;
      b = new_alpha;
    }
    // #ifdef augustdebug
    //     std::cout << iterations << ' ' << a<<' '<<b<<' '<<std::abs(b-a)<<' '<< source_orientation << ' '<< mid_orientation<<' ' << tools::plot((1-(a+b)/2)*source + (a+b)/2*target)<<' '<<tools::plot((1-a)*source+a*target)<<std::endl;
    // #endif
      iterations++;
  }

  if (is_orientation_set)
  {
    // Does it matter if we take distinguish between a == b or
    // (a+b)/2. Prioritize preferred_orientation.
    if (mid_orientation == 0)
    {
      std::cout << mid_orientation;
      PPause;
    }
    else
    {
      Point za = (1-a)*source + a*target;
      Point zb = (1-b)*source + b*target;
      Point zalpha = (1-new_alpha)*source + new_alpha*target;
      const double ab = (a+b)/2;
      Point zab = (1-ab)*source + ab*target;
      const double za_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), za.coordinates());
      const double zb_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), zb.coordinates());
      const double zalpha_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), zalpha.coordinates());
      const double zab_orientation = orient2d(ref_source.coordinates(), ref_target.coordinates(), zab.coordinates());
#ifdef augustdebug
      std::cout << orientation<<"     "<<za_orientation<<' '<<zb_orientation<<' '<<zalpha_orientation<<' '<<zab_orientation<<'\n';
#endif

      if (std::signbit(za_orientation) == orientation)
      {
	z = za;
      }
      else if (std::signbit(zb_orientation) == orientation)
      {
	z = zb;
      }
      else if (std::signbit(zalpha_orientation) == orientation)
      {
	z = zalpha;
      }
      else if (std::signbit(zab_orientation) == orientation)
      {
	z = zab;
      }
      else if (std::signbit(mid_orientation) == orientation)
      {
	z = new_point;
      }
      else
      {
#ifdef augustdebug
	// need to look at history
	std::cout << za_orientation<<' '<<zab_orientation << ' '<<orientation<<std::endl;
#endif
	PPause;
      }
    }
  }
  else
  {
    if (a == b)
    {
      //intersection.push_back(source + a*r);
      z = (1-a)*source + a*target;
#ifdef augustdebug
      std::cout << "        Case 3 with bisection equal:\n"
		<< "        " <<tools::plot(z,"'mo'") << std::endl;
#endif
    }
    else
    {
      //intersection.push_back(source + (a+b)/2*r);
      //const double new_alpha = (a+b)/2;
      z = ((2-a-b)*source + (a+b)*target) / 2;
#ifdef augustdebug
      std::cout << "        Case 3 with bisection half half:\n"
		<< "        " <<tools::plot(z,"'mo'") << std::endl;
#endif
    }
  }

  // Check solution
  if (check_solution)
  {

#ifdef augustdebug
    for (std::size_t d = 0; d < 2; ++d)
      std::cout << CollisionPredicates::collides_segment_point_1d(source[d],target[d],z[d])<<' ';
    std::cout << std::endl;
    for (std::size_t d = 0; d < 2; ++d)
      std::cout << CollisionPredicates::collides_segment_point_1d(ref_source[d],ref_target[d],z[d])<<' ';
    std::cout << std::endl;
    std::cout << CollisionPredicates::collides_segment_point_2d(source, target, z)<<' '<<CollisionPredicates::collides_segment_point_2d(ref_source, ref_target, z)<<std::endl;
#endif
    const bool all_inside =
      ((CollisionPredicates::collides_segment_point_1d(source[0], target[0], z[0]) and
	CollisionPredicates::collides_segment_point_1d(source[1], target[1], z[1]) and
	CollisionPredicates::collides_segment_point_1d(ref_source[0], ref_target[0], z[0]) and
	CollisionPredicates::collides_segment_point_1d(ref_source[1], ref_target[1], z[1])) or
       (CollisionPredicates::collides_segment_point_2d(source, target, z) and
	CollisionPredicates::collides_segment_point_2d(ref_source, ref_target, z)));
    if (all_inside)
    {
      return true;
    }

    // Check that z is inside the smallest interval
    std::vector<std::pair<double,Point> > dists(4);
    dists[0] = std::pair<double,Point>(z.squared_distance(source), source);
    dists[1] = std::pair<double,Point>(z.squared_distance(target), target);
    dists[2] = std::pair<double,Point>(z.squared_distance(ref_source), ref_source);
    dists[3] = std::pair<double,Point>(z.squared_distance(ref_target), ref_target);
    std::sort(dists.begin(), dists.end(), [](std::pair<double,Point> left,
					     std::pair<double,Point> right) {
		return left.first < right.first;
	      });
    const Point r = source.squared_distance(target) > ref_source.squared_distance(ref_target) ? source - target : ref_source - ref_target;
    const double t0 = r.dot(z-dists[0].second);
    const double t1 = r.dot(z-dists[1].second);
    const double t2 = r.dot(z-dists[2].second);
    const double t3 = r.dot(z-dists[3].second);
    std::size_t cnt = 0;
    if (t0<0) cnt++;
    if (t1<0) cnt++;
    if (t2<0) cnt++;
    if (t3<0) cnt++;
#ifdef augustdebug
    std::cout << t0<<' '<<t1<<' '<<t2<<' '<<t3<<std::endl;
    std::cout << dists[0].first<<' '<<dists[1].first <<' '<<dists[2].first<<' '<<dists[3].first<<std::endl;
#endif

    if (cnt == 2 or
	(std::abs(t0) < DOLFIN_EPS and std::abs(t1) < DOLFIN_EPS and
	 std::abs(t2) < DOLFIN_EPS and std::abs(t3) < DOLFIN_EPS))
    {
      return true;
    }

    return false;
  }
  else
  {
    return true;
  }
}
//-----------------------------------------------------------------------------
