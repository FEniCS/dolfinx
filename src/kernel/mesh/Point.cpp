// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-12
// Last changed: 2006-10-16

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
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Point& p)
{
   stream << "[ Point x = " << p.x() << " y = " << p.y() << " z = " << p.z() << " ]";
   return stream;
}
//-----------------------------------------------------------------------------
