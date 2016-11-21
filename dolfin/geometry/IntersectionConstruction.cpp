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
// Last changed: 2016-11-21

#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/math/basic.h>
#include "predicates.h"
#include "GeometryDebugging.h"
#include "CollisionPredicates.h"
#include "IntersectionConstruction.h"


// FIXME august
#include <ttmath/ttmath.h>
#include </home/august/dolfin_simplex_tools.h>


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
    return intersection_segment_segment_1d(p0[0], p1[0], q0[0], q1[0]);
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
std::vector<Point>
IntersectionConstruction::_intersection_segment_segment_1d(double p0,
							   double p1,
							   double q0,
							   double q1)
{
  dolfin_error("IntersectionConstruction.cpp",
	       "compute segment-segment intersection",
	       "Not implemented for dimension 1.");

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
std::vector<Point>
IntersectionConstruction::_intersection_segment_segment_2d(Point p0,
							   Point p1,
							   Point q0,
							   Point q1)
{
  std::vector<Point> intersection;

  // Add vertex-vertex collision to the intersection
  if (p0 == q0 or p0 == q1)
    intersection.push_back(p0);
  if (p1 == q0 or p1 == q1)
    intersection.push_back(p1);

  // Add vertex-"segment interior" collisions to the intersection
  if (CollisionPredicates::collides_interior_point_segment_2d(q0, q1, p0))
    intersection.push_back(p0);

  if (CollisionPredicates::collides_interior_point_segment_2d(q0, q1, p1))
    intersection.push_back(p1);

  if (CollisionPredicates::collides_interior_point_segment_2d(p0, p1, q0))
    intersection.push_back(q0);

  if (CollisionPredicates::collides_interior_point_segment_2d(p0, p1, q1))
    intersection.push_back(q1);

  if (intersection.empty())
  {
    // No collisions in any vertices, so check interior
    return _intersection_segment_interior_segment_interior_2d(p0, p1, q0, q1);
  }

  return intersection;

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
      // std::cout << __FUNCTION__ << "points " << p0<<' '<<p1<<' '<<q0<<' '<<q1<<'\n'
      // 		<< " parallel numerator/denomenator="<<numerator<<" / " << denom << " = " << alpha << std::endl;
      // std::cout << tools::drawtriangle({p0,p1})<<tools::drawtriangle({q0,q1})<<std::endl;


      // Test exact arithmetic for the denominator
      {
	typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(128)> TT;
	const TT p0x(p0.x()), p0y(p0.y()), p1x(p1.x()), p1y(p1.y()), q0x(q0.x()), q0y(q0.y()), q1x(q1.x()), q1y(q1.y()), numerator_tt(numerator);
	const TT denom_tt = (p1.x()-p0.x())*(q1.y()-q0.y()) - (p1.y()-p0.y())*(q1.x()-q0.x());

	// std::cout <<"denom zero? " << (denom_tt == 0) << " p0==p1 " << (p0==p1)<<" q0 == q1 " << (q0==q1)<<" dists " << (p1-p0).squared_norm() << ' ' << (q1-q0).squared_norm() << std::endl;

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
	  // std::cout << "% intersection(s):\n";
	  // for (const Point p: intersection)
	  //   std::cout << tools::plot(p,"'gx'");
	  // std::cout << std::endl;


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
      }

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
    points.insert(points.end(), intersection.begin(), intersection.end());
  }

  if (CollisionPredicates::collides_segment_segment_2d(p0, p2, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_2d(p0, p2, q0, q1);
    points.insert(points.end(), intersection.begin(), intersection.end());
  }

  if (CollisionPredicates::collides_segment_segment_2d(p1, p2, q0, q1))
  {
    const std::vector<Point> intersection = intersection_segment_segment_2d(p1, p2, q0, q1);
    points.insert(points.end(), intersection.begin(), intersection.end());
  }

  // Remove strict duplictes. Use exact equality here. Approximate
  // equality is for ConvexTriangulation.
  // FIXME: This can be avoided if we use interior segment tests.
  std::vector<Point> unique_points;

  for (std::size_t i = 0; i < points.size(); ++i)
  {
    bool unique = true;
    for (std::size_t j = i+1; j < points.size(); ++j)
    {
      if (points[i] == points[j])
      {
	unique = false;
	break;
      }
    }
    if (unique)
      unique_points.push_back(points[i]);
  }

  return unique_points;
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
IntersectionConstruction::_intersection_triangle_triangle_2d(const Point& p0,
							     const Point& p1,
							     const Point& p2,
							     const Point& q0,
							     const Point& q1,
							     const Point& q2)
{
  std::vector<dolfin::Point> points;

  if (CollisionPredicates::collides_triangle_triangle_2d(p0, p1, p2,
							 q0, q1, q2))
  {
    // Pack points as vectors
    std::array<Point, 3> tri_0({p0, p1, p2});
    std::array<Point, 3> tri_1({q0, q1, q2});

    // Extract coordinates
    double t0[3][2] = {{p0[0], p0[1]}, {p1[0], p1[1]}, {p2[0], p2[1]}};
    double t1[3][2] = {{q0[0], q0[1]}, {q1[0], q1[1]}, {q2[0], q2[1]}};

    // Find all vertex-vertex collision
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = 0; j < 3; j++)
      {
	if (tri_0[i] == tri_1[j])
	  points.push_back(tri_0[i]);
      }
    }

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

    // Find all "edge interior"-"edge interior" intersections
    for (std::size_t i = 0; i < 3; i++)
    {
      for (std::size_t j = 0; j < 3; j++)
      {
	{
	  std::vector<Point> triangulation =
	    intersection_segment_interior_segment_interior_2d(tri_0[i],
							      tri_0[(i+1)%3],
							      tri_1[j],
							      tri_1[(j+1)%3]);
	  points.insert(points.end(), triangulation.begin(), triangulation.end());
	}
      }
    }

    // Find alle vertex-"triangle interior" intersections
    const int s0 = std::signbit(orient2d(t0[0], t0[1], t0[2])) == true ? -1 : 1;
    const int s1 = std::signbit(orient2d(t1[0], t1[1], t1[2])) == true ? -1 : 1;

    for (std::size_t i = 0; i < 3; ++i)
    {
      const double q0_q1_pi = s1*orient2d(t1[0], t1[1], t0[i]);
      const double q1_q2_pi = s1*orient2d(t1[1], t1[2], t0[i]);
      const double q2_q0_pi = s1*orient2d(t1[2], t1[0], t0[i]);

      if (q0_q1_pi > 0. and
	  q1_q2_pi > 0. and
	  q2_q0_pi > 0.)
      {
	points.push_back(tri_0[i]);
      }

      const double p0_p1_qi = s0*orient2d(t0[0], t0[1], t1[i]);
      const double p1_p2_qi = s0*orient2d(t0[1], t0[2], t1[i]);
      const double p2_p0_qi = s0*orient2d(t0[2], t0[0], t1[i]);

      if (p0_p1_qi > 0. and
	  p1_p2_qi > 0. and
	  p2_p0_qi > 0.)
      {
	points.push_back(tri_1[i]);
      }
    }

  }

  return points;
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
