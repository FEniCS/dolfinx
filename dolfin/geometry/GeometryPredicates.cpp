// Copyright (C) 2016 Anders Logg, August Johansson and Benjamin Kehlet
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
// Last changed: 2017-03-01

#include <cmath>
#include "GeometryPredicates.h"
#include "predicates.h"

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
    dolfin_debug("check");
    return is_degenerate_3d(simplex);
  default:
    dolfin_error("GeometryPredicates.cpp",
		 "is_degenerate",
		 "Unkonwn dimension (only implemented for dimension 2 and 3");
  }
  return false;
}
//-----------------------------------------------------------------------------
namespace
{
  bool operator==(Point a, Point b)
  {
    return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
  }
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_2d(std::vector<Point> simplex)
{
  if (simplex.size() < 2 or simplex.size() > 3)
  {
    info("Degenerate 2D simplex with %d vertices.", simplex.size());
    return true;
  }

  switch (simplex.size())
  {
  case 2: return simplex[0] == simplex[1];
  case 3: return orient2d(simplex[0], simplex[1], simplex[2]) == 0;
  }

  // Shouldn't get here
  dolfin_error("CGALExactArithmetic.h",
               "call _is_degenerate_2d",
               "Only implemented for simplices of tdim 0, 1 and 2, not tdim = %d",
               simplex.size() - 1);

  return true;
}
//------------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_3d(std::vector<Point> simplex)
{
  if (simplex.size() < 2 or simplex.size() > 4)
  {
    info("Degenerate 3D simplex with %d vertices.", simplex.size());
    return true;
  }

  switch (simplex.size())
  {
  case 2: return simplex[0] == simplex[1];
  case 3:
    {
      const double ayz[2] = {simplex[0].y(), simplex[0].z()};
      const double byz[2] = {simplex[1].y(), simplex[1].z()};
      const double cyz[2] = {simplex[2].y(), simplex[2].z()};
      if (_orient2d(ayz, byz, cyz) != 0.)
	return false;

      const double azx[2] = {simplex[0].z(), simplex[0].x()};
      const double bzx[2] = {simplex[1].z(), simplex[1].x()};
      const double czx[2] = {simplex[2].z(), simplex[2].x()};
      if (_orient2d(azx, bzx, czx) != 0.)
	return false;

      const double axy[2] = {simplex[0].x(), simplex[0].y()};
      const double bxy[2] = {simplex[1].x(), simplex[1].y()};
      const double cxy[2] = {simplex[2].x(), simplex[2].y()};
      if (_orient2d(axy, bxy, cxy) != 0.)
	return false;

      return true;
    }
  case 4: return orient3d(simplex[0], simplex[1], simplex[2], simplex[3]) == 0;
  }

  // Shouldn't get here
  dolfin_error("CGALExactArithmetic.h",
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
    if (!std::isfinite(p.x())) return false;
    if (!std::isfinite(p.y())) return false;
    if (!std::isfinite(p.z())) return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_finite(const std::vector<double>& simplex)
{
  for (double p : simplex)
  {
    if (!std::isfinite(p)) return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
