// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2006-10-23

#include <cmath>
#include <dolfin/Point.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
real Point::distance(const Point& p) const
{
  const real dx = p._x[0] - _x[0];
  const real dy = p._x[1] - _x[1];
  const real dz = p._x[2] - _x[2];

  return std::sqrt(dx*dx + dy*dy + dz*dz);
}
//-----------------------------------------------------------------------------
real Point::norm() const
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
real Point::dot(const Point& p) const
{
  return _x[0]*p._x[0] + _x[1]*p._x[1] + _x[2]*p._x[2];
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Point& p)
{
   stream << "[ Point x = " << p.x() << " y = " << p.y() << " z = " << p.z() << " ]";
   return stream;
}
//-----------------------------------------------------------------------------
