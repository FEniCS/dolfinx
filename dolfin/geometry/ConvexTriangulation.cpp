// Copyright (C) 2016 Anders Logg and August Johansson, Benjamin Kehlet
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
// First added:  2016-06-01
// Last changed: 2016-06-05

#include "ConvexTriangulation.h"
#include "predicates.h"
#include <algorithm>


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

//------------------------------------------------------------------------------
std::vector<std::vector<Point>>
ConvexTriangulation::triangulate(std::vector<Point> p,
                                 std::size_t gdim,
                                 std::size_t tdim)
{
  if (tdim == 2 && gdim == 2)
    return triangulate_graham_scan(p, gdim);

  if (tdim == 1)
  {
    // FIXME: Is this correct?
    if (p.size() > 2)
      dolfin_error("ConvexTriangulation.cpp",
                   "triangulate convex polyhedron",
                   "a convex polyhedron of topological dimension 1 can not have more then 2 points");
    std::vector<std::vector<Point>> t;
    t.push_back(p);
    return t;
  }

  dolfin_error("ConvexTriangulation.cpp",
               "triangulate convex polyhedron",
               "triangulation of polyhedron of topological dimension %u and geometric dimension %u not implemented", tdim, gdim);
}
//------------------------------------------------------------------------------
std::vector<std::vector<Point>>
ConvexTriangulation::triangulate_graham_scan(std::vector<Point> points,
                                             std::size_t gdim)
{
  if (points.size() < 3)
    return std::vector<std::vector<Point>>();

  std::vector<std::vector<Point>> triangulation;

  if (points.size() == 3)
  {
    triangulation.push_back(points);
    return triangulation;
  }

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
  triangulation.reserve(order.size() - 1);
  for (std::size_t m = 0; m < order.size()-1; ++m)
  {
    // FIXME: We could consider only triangles with area > tolerance here.
    triangulation.push_back(
      { points[0],
          points[order[m].second],
         points[order[m + 1].second] });
  }

  return triangulation;
}
//-----------------------------------------------------------------------------
 std::vector<std::vector<Point>> triangulate_3d(std::vector<Point> points)
 {

   // Tolerance for coplanar points.
   // How is this chosen?
   const double coplanar_tol = 1000*DOLFIN_EPS_LARGE;

   // Points forming the tetrahedral partitioning of the polyhedron
   std::vector<std::vector<Point>> triangulation;

  if (points.size() == 4)
  {
    // FIXME: Check that the volume is sufficiently large
    triangulation.push_back(points);
  }
  else if (points.size() > 4)
  {

    // Tetrahedra are created using the facet points and a center point.
    Point polyhedroncenter = points[0];
    for (std::size_t i = 1; i < points.size(); ++i)
      polyhedroncenter += points[i];
    polyhedroncenter /= points.size();

    // Data structure for storing checked triangle indices (do this
    // better with some fancy stl structure?)
    const std::size_t N = points.size(), N2 = points.size()*points.size();

    // FIXME: this is expensive
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
  }
  return triangulation;
 }


//-----------------------------------------------------------------------------
bool ConvexTriangulation::_is_degenerate(std::vector<Point> s)
{
  bool is_degenerate = false;

  switch (s.size())
  {
  case 0:
    // FIXME: Is this correct? Is "nothing" degenerate?
    is_degenerate = true;
    break;
  case 1:
    /// FIXME: Is this correct? Can a point be degenerate?
    is_degenerate = true;
    break;
  case 2:
    {
      is_degenerate = s[0]==s[1];
      // FIXME: verify with orient2d
      // double r[2] = { dolfin::rand(), dolfin::rand() };
      // is_degenerate = orient2d(s[0].coordinates(), s[1].coordinates(), r) == 0;

      // // FIXME: compare with ==
      // dolfin_assert(is_degenerate == (s[0] == s[1]));

      break;
    }
  case 3:
    is_degenerate = orient2d(s[0].coordinates(),
			     s[1].coordinates(),
			     s[2].coordinates()) == 0;
    break;
  default:
    dolfin_error("ConvexTriangulation.cpp",
		 "_is_degenerate",
		 "Only implemented for simplices of tdim 0, 1 and 2");
  }

  return is_degenerate;
}
//------------------------------------------------------------------------------
