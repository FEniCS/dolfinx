// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Legendre.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Legendre::Legendre(int n)
{
  if ( n < 0 )
    dolfin_error("Degree for Legendre polynomial must be non-negative.");

  this->n = n;
}
//-----------------------------------------------------------------------------
real Legendre::operator() (real x)
{
  return eval(n, x);
}
//-----------------------------------------------------------------------------
real Legendre::ddx(real x)
{
  return ddx(n, x);
}
//-----------------------------------------------------------------------------
real Legendre::d2dx(real x)
{
  return d2dx(n, x);
}
//-----------------------------------------------------------------------------
real Legendre::eval(int n, real x)
{
  // Special case n = 0
  if ( n == 0 )
    return 1.0;

  // Special case n = 1
  if ( n == 1 )
    return x;
  
  // Recurrence, BETA page 254
  real nn = real(n);
  return ( (2.0*nn-1.0)*x*eval(n-1, x) - (nn-1.0)*eval(n-2, x) ) / nn;
}
//-----------------------------------------------------------------------------
real Legendre::ddx(int n, real x)
{
  // Special case n = 0
  if ( n == 0 )
    return 0.0;
  
  // Special case n = 1
  if ( n == 1 )
    return 1.0;
  
  // Avoid division by zero
  if ( fabs(x - 1.0) < DOLFIN_EPS )
    x -= 2.0*DOLFIN_EPS;
  if ( fabs(x + 1.0) < DOLFIN_EPS )
    x += 2.0*DOLFIN_EPS;
  
  // Formula, BETA page 254
  real nn = real(n);
  return nn * (x*eval(n, x) - eval(n-1, x)) / (x*x - 1.0);
}
//-----------------------------------------------------------------------------
real Legendre::d2dx(int, real x)
{
  // Special case n = 0
  if ( n == 0 )
    return 0.0;

  // Special case n = 1
  if ( n == 1 )
    return 0.0;

  // Avoid division by zero
  if ( fabs(x - 1.0) < DOLFIN_EPS )
    x -= 2.0*DOLFIN_EPS;
  if ( fabs(x + 1.0) < DOLFIN_EPS )
    x += 2.0*DOLFIN_EPS;

  // Formula, BETA page 254
  real nn = real(n);
  return ( 2.0*x*ddx(n, x) - nn*(nn+1)*eval(n, x) ) / (1.0-x*x);
}
//-----------------------------------------------------------------------------
