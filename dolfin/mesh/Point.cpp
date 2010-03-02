// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2010-02-28

#include <cmath>
#include "Point.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Point::distance(const Point& p) const
{
  const double dx = p._x[0] - _x[0];
  const double dy = p._x[1] - _x[1];
  const double dz = p._x[2] - _x[2];

  return std::sqrt(dx*dx + dy*dy + dz*dz);
}
//-----------------------------------------------------------------------------
double Point::norm() const
{
  return std::sqrt(_x[0]*_x[0] + _x[1]*_x[1] + _x[2]*_x[2]);
}
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
double Point::dot(const Point& p) const
{
  return _x[0]*p._x[0] + _x[1]*p._x[1] + _x[2]*p._x[2];
}
//-----------------------------------------------------------------------------
std::string Point::str(bool verbose) const
{
  std::stringstream s;
  s << "<Point x = " << x() << " y = " << y() << " z = " << z() << ">";
  return s.str();
}
//-----------------------------------------------------------------------------

