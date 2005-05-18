// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <cmath>
#include <dolfin/Point.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Point::Point() : x(0.0), y(0.0), z(0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Point::Point(real x) : x(x), y(0.0), z(0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Point::Point(real x, real y) : x(x), y(y), z(0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Point::Point(real x, real y, real z) : x(x), y(y), z(z)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Point::Point(const Point& p) : x(p.x), y(p.y), z(p.z)
{
  // Do nothing
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

  return sqrt(dx*dx + dy*dy + dz*dz);
}
//-----------------------------------------------------------------------------
real Point::norm() const
{
  return sqrt(x*x + y*y + z*z);
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
Point::operator real() const
{
  return x;
}
//-----------------------------------------------------------------------------
const Point& Point::operator= (real x)
{
  this->x = x;
  this->y = 0.0;
  this->z = 0.0;

  return *this;
}
//-----------------------------------------------------------------------------
Point Point::operator+ (const Point& p) const
{
  Point q(*this);
  q += p;  
  return q;
}
//-----------------------------------------------------------------------------
Point Point::operator- (const Point& p) const
{
  Point q(*this);
  q -= p;
  return q;
}
//-----------------------------------------------------------------------------
real Point::operator* (const Point& p) const
{
  return x*p.x + y*p.y + z*p.z;
}
//-----------------------------------------------------------------------------
const Point& Point::operator+= (const Point& p)
{
  x += p.x;
  y += p.y;
  z += p.z;

  return *this;
}
//-----------------------------------------------------------------------------
const Point& Point::operator-= (const Point& p)
{
  x -= p.x;
  y -= p.y;
  z -= p.z;

  return *this;
}
//-----------------------------------------------------------------------------
const Point& Point::operator*= (real a)
{
  x *= a;
  y *= a;
  z *= a;

  return *this;
}
//-----------------------------------------------------------------------------
const Point& Point::operator/= (real a)
{
  x /= a;
  y /= a;
  z /= a;

  return *this;
}
//-----------------------------------------------------------------------------
Point Point::cross(const Point& p) const
{
  Point q;

  q.x = y*p.z - z*p.y;
  q.y = z*p.x - x*p.z;
  q.z = x*p.y - y*p.x;

  return q;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Point& p)
{
  stream << "[ Point x = " << p.x << " y = " << p.y << " z = " << p.z << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
