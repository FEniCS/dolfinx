#include <dolfin/shapefunctions.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// Linear shape functions on the reference triangle
//-----------------------------------------------------------------------------
real dolfin::tri10(real x, real y, real z, real t)
{
  return 1 - x - y;
}
//-----------------------------------------------------------------------------
real dolfin::tri11(real x, real y, real z, real t)
{
  return x;
}
//-----------------------------------------------------------------------------
real dolfin::tri12(real x, real y, real z, real t)
{
  return y;
}
//-----------------------------------------------------------------------------
// Linear shape functions on the reference tetrahedron
//-----------------------------------------------------------------------------
real dolfin::tet10(real x, real y, real z, real t)
{
  return 1 - x - y - z;
}
//-----------------------------------------------------------------------------
real dolfin::tet11(real x, real y, real z, real t)
{
  return x;
}
//-----------------------------------------------------------------------------
real dolfin::tet12(real x, real y, real z, real t)
{
  return y;
}
//-----------------------------------------------------------------------------
real dolfin::tet13(real x, real y, real z, real t)
{
  return z;
}
//-----------------------------------------------------------------------------
