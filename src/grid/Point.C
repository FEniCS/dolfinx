#include "Point.hh"
#include <math.h>

//-----------------------------------------------------------------------------
real Point::Distance(Point p)
{
  real dx = real(x) - real(p.x);
  real dy = real(y) - real(p.y);
  real dz = real(z) - real(p.z);

  return ( sqrt(dx*dx+dy*dy+dz*dz) );
}
//-----------------------------------------------------------------------------
