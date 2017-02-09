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
// Last changed: 2017-02-09

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
bool GeometryPredicates::_is_degenerate_2d(std::vector<Point> simplex)
{
  bool is_degenerate = false;

  switch (simplex.size())
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
      is_degenerate = (simplex[0] == simplex[1]);
      // FIXME: verify with orient2d
      // double r[2] = { dolfin::rand(), dolfin::rand() };
      // is_degenerate = orient2d(s[0].coordinates(), s[1].coordinates(), r) == 0;

      // // FIXME: compare with ==
      // dolfin_assert(is_degenerate == (s[0] == s[1]));

      break;
    }
  case 3:
    is_degenerate = orient2d(simplex[0],
			     simplex[1],
			     simplex[2]) == 0;
    break;
  default:
    dolfin_error("GeometryPredicates.cpp",
		 "_is_degenerate_2d",
		 "Only implemented for simplices of tdim 0, 1 and 2.");
  }

  return is_degenerate;
}
//------------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_3d(std::vector<Point> simplex)
{
  dolfin_debug("check");

  switch (simplex.size())
  {
  case 4:
    return orient3d(simplex[0], simplex[1], simplex[2], simplex[3]) == 0;
  default:
    dolfin_error("GeometryPredicates.cpp",
		 "check degeneracy of simplex",
		 "Only implemented for simplices of tdim 3");
  }

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
