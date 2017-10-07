// Copyright (C) 2016-2017 Anders Logg, August Johansson and Benjamin Kehlet
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
// First added:  2016-11-21
// Last changed: 2017-10-07

#include <cmath>
#include "CGALExactArithmetic.h"
#include "predicates.h"
#include "GeometryPredicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool GeometryPredicates::is_degenerate(const std::vector<Point>& simplex,
                                       std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return is_degenerate_2d(simplex);
  case 3:
    return is_degenerate_3d(simplex);
  default:
    dolfin_error("GeometryPredicates.cpp",
		 "is_degenerate",
		 "Unkonwn dimension (only implemented for dimension 2 and 3");
  }
  return false;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_2d(const std::vector<Point>& simplex)
{
  switch (simplex.size())
  {
  case 2:
    return simplex[0] == simplex[1];
  case 3:
    return orient2d(simplex[0], simplex[1], simplex[2]) == 0.0;
  }

  // Shouldn't get here
  dolfin_error("GeometryPredicates.h",
               "call _is_degenerate_2d",
               "Only implemented for simplices of tdim 0, 1 and 2, not tdim = %d",
               simplex.size() - 1);

  return true;
}
//------------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_3d(const std::vector<Point>& simplex)
{
  switch (simplex.size())
  {
  case 2:
    return simplex[0] == simplex[1];
  case 3:
    {
      const double ayz[2] = {simplex[0].y(), simplex[0].z()};
      const double byz[2] = {simplex[1].y(), simplex[1].z()};
      const double cyz[2] = {simplex[2].y(), simplex[2].z()};
      if (_orient2d(ayz, byz, cyz) != 0.0)
	return false;

      const double azx[2] = {simplex[0].z(), simplex[0].x()};
      const double bzx[2] = {simplex[1].z(), simplex[1].x()};
      const double czx[2] = {simplex[2].z(), simplex[2].x()};
      if (_orient2d(azx, bzx, czx) != 0.0)
	return false;

      const double axy[2] = {simplex[0].x(), simplex[0].y()};
      const double bxy[2] = {simplex[1].x(), simplex[1].y()};
      const double cxy[2] = {simplex[2].x(), simplex[2].y()};
      if (_orient2d(axy, bxy, cxy) != 0.0)
	return false;

      return true;
    }
  case 4:
    return orient3d(simplex[0], simplex[1], simplex[2], simplex[3]) == 0.0;
  }

  // Shouldn't get here
  dolfin_error("GeometryPredicates.h",
               "call _is_degenerate_3d",
               "Only implemented for simplices of tdim 0, 1, 2 and 3, not tdim = %d",
               simplex.size() - 1);

  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_finite(const std::vector<Point>& simplex)
{
  for (auto p : simplex)
  {
    if (!std::isfinite(p.x()))
      return false;
    if (!std::isfinite(p.y()))
      return false;
    if (!std::isfinite(p.z()))
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_finite(const std::vector<double>& simplex)
{
  for (double p : simplex)
  {
    if (!std::isfinite(p))
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::convex_hull_is_degenerate(const std::vector<Point>& points,
                                                   std::size_t gdim)
{
  // Points are assumed to be unique

  if (points.size() < gdim+1)
    return true;

  if (gdim == 2)
  {
    // FIXME!
    return false;
  }
  else if (gdim == 3)
  {
    int i = 0, j = 1, k = 2;
    bool found = false;

    // Find three point which are not collinear
    for (; i < points.size(); i++)
    {
      for (j = i+1; j < points.size(); j++)
      {
        for (k = j+1; k < points.size(); k++)
        {
          const Point ij = points[j] - points[i];
          const Point ik = points[k] - points[i];
          if ( -(std::abs( (ij/ij.norm() ).dot(ik/ik.norm()))-1)  > DOLFIN_EPS)
          {
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (found)
        break;
    }

    // All points are collinear
    if (!found)
      return false;

    for (int l = 0; l < points.size();  l++)
    {
      if (l == i || l == j || l == k)
        continue;

      if (orient3d(points[i], points[j], points[k], points[l]) == 0.0)
        return true;
    }

    return false;
  }
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_degenerate_2d(const std::vector<Point>& simplex)
{
  return CHECK_CGAL(_is_degenerate_2d(simplex),
		    cgal_is_degenerate_2d(simplex));
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_degenerate_3d(const std::vector<Point>& simplex)
{
  return CHECK_CGAL(_is_degenerate_3d(simplex),
		    cgal_is_degenerate_3d(simplex));
}
