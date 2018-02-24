// Copyright (C) 2006-2008 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Point.h"
#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
const Point Point::cross(const Point& p) const
{
  Point q;
  q._x[0] = _x[1] * p._x[2] - _x[2] * p._x[1];
  q._x[1] = _x[2] * p._x[0] - _x[0] * p._x[2];
  q._x[2] = _x[0] * p._x[1] - _x[1] * p._x[0];

  return q;
}
//-----------------------------------------------------------------------------
double Point::squared_distance(const Point& p) const
{
  const double dx = p._x[0] - _x[0];
  const double dy = p._x[1] - _x[1];
  const double dz = p._x[2] - _x[2];

  return dx * dx + dy * dy + dz * dz;
}
//-----------------------------------------------------------------------------
double Point::dot(const Point& p) const
{
  return _x[0] * p._x[0] + _x[1] * p._x[1] + _x[2] * p._x[2];
}
//-----------------------------------------------------------------------------
Point Point::rotate(const Point& k, double theta) const
{
  dolfin_assert(std::abs(k.norm() - 1.0) < DOLFIN_EPS);

  const Point& v = *this;
  const double cosTheta = cos(theta);
  const double sinTheta = sin(theta);

  // Rodriques' rotation formula
  return v * cosTheta + k.cross(v) * sinTheta + k * k.dot(v) * (1 - cosTheta);
}
//-----------------------------------------------------------------------------
std::string Point::str(bool verbose) const
{
  std::stringstream s;
  s << "<Point x = " << x() << " y = " << y() << " z = " << z() << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
