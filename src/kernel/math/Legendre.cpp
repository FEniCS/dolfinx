// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Legendre.h>

//-----------------------------------------------------------------------------
Legendre::Legendre(int n)
{
  if ( n < 0 )
    dolfin_error("Degree for Legendre polynomial must be non-negative.");

  this->n = n;
}
//-----------------------------------------------------------------------------
real Legendre::eval(real x)
{
  // Special case n = 0
  if ( n == 0 )
    return 1.0;

  // Special case n = 1
  if ( n == 1 )
    return x;
  
  // Recurrence, BETA page 254
  real nn = real(n);
  return ( (2.0*nn-1.0)*x*eval(n-1,x) - (nn-1.0)*eval(n-2,x) ) / nn;
}
//-----------------------------------------------------------------------------
real Legendre::operator(real x)
{
  return eval(x, n);
}
//-----------------------------------------------------------------------------
real Legendre::dx(real x)
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
  return nn * (x*eval(n,x) - eval(n-1,x)) / (x*x - 1.0);
}
//-----------------------------------------------------------------------------
real Legendre::dx2(real x)
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
  return ( 2.0*x*dx(n,x) - nn*(nn+1)*eval(n,x) ) / (1.0-x*x);
}
//-----------------------------------------------------------------------------
