// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/Point.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Point::Point()
{
  x = 0.0;
  y = 0.0;
  z = 0.0;
}
//-----------------------------------------------------------------------------
Point::Point(real x)
{
  this->x = x;
  y = 0.0;
  z = 0.0;
}
//-----------------------------------------------------------------------------
Point::Point(real x, real y)
{
  this->x = x;
  this->y = y;
  z = 0.0;
}
//-----------------------------------------------------------------------------
Point::Point(real x, real y, real z)
{
  this->x = x;
  this->y = y;
  this->z = z;
}
//-----------------------------------------------------------------------------
real Point::dist(Point p) const
{
  return dist(p.x, p.y, p.z);
}
//-----------------------------------------------------------------------------
real Point::dist(real x, real y, real z) const
{
  real dx = x - this->x;
  real dy = y - this->y;
  real dz = z - this->z;

  return sqrt( dx*dx + dy*dy + dz*dz );
}
//-----------------------------------------------------------------------------
Point Point::midpoint(Point p) const
{
  real mx = 0.5*(x + p.x);
  real my = 0.5*(y + p.y);
  real mz = 0.5*(z + p.z);

  Point mp(mx,my,mz);

  return mp;
}
//-----------------------------------------------------------------------------
void Point::operator= (real x)
{
  this->x = x;
  this->y = 0.0;
  this->z = 0.0;
}
//-----------------------------------------------------------------------------
Point::operator real() const
{
  return x;
}
//-----------------------------------------------------------------------------
Point Point::operator+= (const Point& p)
{
  x += p.x;
  y += p.y;
  z += p.z;

  return *this;
}
//-----------------------------------------------------------------------------
Point Point::operator/= (real a)
{
  x /= a;
  y /= a;
  z /= a;

  return *this;
}
//-----------------------------------------------------------------------------
// Output
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Point& p)
{
  stream << "[ Point x = " << p.x << " y = " << p.y << " z = " << p.z << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
