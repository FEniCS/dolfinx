// Copyright (C) 2006-2008 Anders Logg
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
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2014-01-06

#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/math/basic.h>
#include "Point.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const Point Point::cross(const Point& p) const
{
  Point q;

  q._x[0] = _x[1]*p._x[2] - _x[2]*p._x[1];
  q._x[1] = _x[2]*p._x[0] - _x[0]*p._x[2];
  q._x[2] = _x[0]*p._x[1] - _x[1]*p._x[0];

  return q;
}
//-----------------------------------------------------------------------------
double Point::squared_distance(const Point& p) const
{
  const double dx = p._x[0] - _x[0];
  const double dy = p._x[1] - _x[1];
  const double dz = p._x[2] - _x[2];

  return dx*dx + dy*dy + dz*dz;
}
//-----------------------------------------------------------------------------
double Point::dot(const Point& p) const
{
  return _x[0]*p._x[0] + _x[1]*p._x[1] + _x[2]*p._x[2];
}
//-----------------------------------------------------------------------------
Point Point::rotate(const Point& k, double theta) const
{
  dolfin_assert(near(k.norm(), 1.0));

  const Point& v = *this;
  const double cosTheta = cos(theta);
  const double sinTheta = sin(theta);

  //Rodriques' rotation formula
  return v*cosTheta + k.cross(v)*sinTheta + k*k.dot(v)*(1-cosTheta);
}
//-----------------------------------------------------------------------------
std::string Point::str(bool verbose) const
{
  std::stringstream s;
  s << "<Point x = " << x() << " y = " << y() << " z = " << z() << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
