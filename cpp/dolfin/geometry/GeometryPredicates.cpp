// Copyright (C) 2016-2017 Anders Logg, August Johansson and Benjamin Kehlet
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cmath>
#include <dolfin/common/constants.h>

#include "CGALExactArithmetic.h"
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
    return is_degenerate_3d(simplex);
  default:
    dolfin_error("GeometryPredicates.cpp", "is_degenerate",
                 "Unkonwn dimension (only implemented for dimension 2 and 3");
  }
  return false;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_2d(const std::vector<Point>& simplex)
{
  if (simplex.size() < 2 or simplex.size() > 3)
  {
    info("Degenerate 2D simplex with %d vertices.", simplex.size());
    return true;
  }

  switch (simplex.size())
  {
  case 2:
    return simplex[0] == simplex[1];
  case 3:
    return orient2d(simplex[0], simplex[1], simplex[2]) == 0.0;
  }

  // Shouldn't get here
  dolfin_error(
      "GeometryPredicates.h", "call _is_degenerate_2d",
      "Only implemented for simplices of tdim 0, 1 and 2, not tdim = %d",
      simplex.size() - 1);

  return true;
}
//------------------------------------------------------------------------------
bool GeometryPredicates::_is_degenerate_3d(const std::vector<Point>& simplex)
{
  if (simplex.size() < 2 or simplex.size() > 4)
  {
    info("Degenerate 3D simplex with %d vertices.", simplex.size());
    return true;
  }

  switch (simplex.size())
  {
  case 2:
    return simplex[0] == simplex[1];
  case 3:
  {
    const double ayz[2] = {simplex[0][1], simplex[0][2]};
    const double byz[2] = {simplex[1][1], simplex[1][2]};
    const double cyz[2] = {simplex[2][1], simplex[2][2]};
    if (_orient2d(ayz, byz, cyz) != 0.0)
      return false;

    const double azx[2] = {simplex[0][2], simplex[0][0]};
    const double bzx[2] = {simplex[1][2], simplex[1][0]};
    const double czx[2] = {simplex[2][2], simplex[2][0]};
    if (_orient2d(azx, bzx, czx) != 0.0)
      return false;

    const double axy[2] = {simplex[0][0], simplex[0][1]};
    const double bxy[2] = {simplex[1][0], simplex[1][1]};
    const double cxy[2] = {simplex[2][0], simplex[2][1]};
    if (_orient2d(axy, bxy, cxy) != 0.0)
      return false;

    return true;
  }
  case 4:
    return orient3d(simplex[0], simplex[1], simplex[2], simplex[3]) == 0.0;
  }

  // Shouldn't get here
  dolfin_error(
      "GeometryPredicates.h", "call _is_degenerate_3d",
      "Only implemented for simplices of tdim 0, 1, 2 and 3, not tdim = %d",
      simplex.size() - 1);

  return true;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_finite(const std::vector<Point>& simplex)
{
  for (auto p : simplex)
  {
    if (!std::isfinite(p[0]))
      return false;
    if (!std::isfinite(p[1]))
      return false;
    if (!std::isfinite(p[2]))
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
bool GeometryPredicates::convex_hull_is_degenerate(
    const std::vector<Point>& points, std::size_t gdim)
{
  // Points are assumed to be unique

  if (points.size() < gdim + 1)
    return true;

  if (gdim == 2)
  {
    // FIXME!
    return false;
  }
  else if (gdim == 3)
  {
    std::size_t i = 0, j = 1, k = 2;
    bool found = false;

    // Find three point which are not collinear
    for (; i < points.size(); i++)
    {
      for (j = i + 1; j < points.size(); j++)
      {
        for (k = j + 1; k < points.size(); k++)
        {
          const Point ij = points[j] - points[i];
          const Point ik = points[k] - points[i];
          if (-(std::abs((ij / ij.norm()).dot(ik / ik.norm())) - 1)
              > DOLFIN_EPS)
          {
            found = true;
            break;
          }
        }
        if (found)
          break;
      }
      if (found)
        break;
    }

    // All points are collinear
    if (!found)
      return false;

    for (std::size_t l = 0; l < points.size(); l++)
    {
      if (l == i || l == j || l == k)
        continue;

      if (orient3d(points[i], points[j], points[k], points[l]) == 0.0)
        return true;
    }

    return false;
  }

  dolfin_error("GeometryPredicates.h", "call convex_hull_is_degenerate",
               "Only fully implemented for gdim == 3, not gdim = %d", gdim);
  return false;
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_degenerate_2d(const std::vector<Point>& simplex)
{
  return CHECK_CGAL(_is_degenerate_2d(simplex), cgal_is_degenerate_2d(simplex));
}
//-----------------------------------------------------------------------------
bool GeometryPredicates::is_degenerate_3d(const std::vector<Point>& simplex)
{
  return CHECK_CGAL(_is_degenerate_3d(simplex), cgal_is_degenerate_3d(simplex));
}
//-----------------------------------------------------------------------------
