#include <dolfin/shapefunctions.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// Linear shape functions on the reference triangle
//-----------------------------------------------------------------------------
real dolfin::trilin0(real x, real y, real z, real t)
{
  return 1 - x - y;
}
//-----------------------------------------------------------------------------
real dolfin::trilin1(real x, real y, real z, real t)
{
  return x;
}
//-----------------------------------------------------------------------------
real dolfin::trilin2(real x, real y, real z, real t)
{
  return y;
}
//-----------------------------------------------------------------------------
// Linear shape functions on the reference tetrahedron
//-----------------------------------------------------------------------------
real dolfin::tetlin0(real x, real y, real z, real t)
{
  return 1 - x - y - z;
}
//-----------------------------------------------------------------------------
real dolfin::tetlin1(real x, real y, real z, real t)
{
  return x;
}
//-----------------------------------------------------------------------------
real dolfin::tetlin2(real x, real y, real z, real t)
{
  return y;
}
//-----------------------------------------------------------------------------
real dolfin::tetlin3(real x, real y, real z, real t)
{
  return z;
}
//-----------------------------------------------------------------------------
