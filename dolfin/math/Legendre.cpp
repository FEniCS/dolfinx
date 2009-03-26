// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-06-03
// Last changed: 2009-02-17

#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/real.h>
#include "Legendre.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Legendre::Legendre(uint n) : n(n), cache_x(0.0)
{
  cache = new real[n+1];
  cache[0] = 1.0; //constant value
  
  //eval to initialize cache
  real tmp = eval(n, -1.0);
  tmp++; //avoid compiler warning
}
//-----------------------------------------------------------------------------
Legendre::~Legendre() 
{
  delete [] cache;
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
real Legendre::eval(uint nn, real x)
{
  //recursive formula, BETA page 254
  //return ( (2.0*nn-1.0)*x*eval(nn-1, x) - (nn-1.0)*eval(nn-2, x) ) / nn;


  //The special cases
  if (n == 0) return 1.0;
  if (n == 1) return x;
  
  //check cache
  if (x != cache_x) 
  {
  
    cache[1] = x;

    for (uint i = 2; i <= n; ++i) 
    {
      real ii(i);
      cache[i] = ( (2.0*ii-1.0)*x*cache[i-1] - (ii-1.0)*cache[i-2] ) / ii;
    }
    cache_x = x;
  }
  
  return cache[nn];
}
//-----------------------------------------------------------------------------
real Legendre::ddx(uint n, real x)
{
  // Special case n = 0
  if (n == 0)
    return 0.0;
  
  // Special case n = 1
  if (n == 1)
    return 1.0;
  
  // Avoid division by zero
  if (abs(x - 1.0) < real_epsilon())
    x -= 2.0*real_epsilon();
  if (abs(x + 1.0) < real_epsilon())
    x += 2.0*real_epsilon();
  
  // Formula, BETA page 254
  real nn = real(n);
  return nn * (x*eval(n, x) - eval(n-1, x)) / (x*x - 1.0);
}
//-----------------------------------------------------------------------------
real Legendre::d2dx(uint, real x)
{
  // Special case n = 0
  if (n == 0)
    return 0.0;

  // Special case n = 1
  if (n == 1)
    return 0.0;

  // Avoid division by zero
  if (abs(x - 1.0) < real_epsilon())
    x -= 2.0*real_epsilon();
  if (abs(x + 1.0) < real_epsilon())
    x += 2.0*real_epsilon();

  // Formula, BETA page 254
  real nn = real(n);
  return (2.0*x*ddx(n, x) - nn*(nn+1)*eval(n, x)) / (1.0-x*x);
}
//-----------------------------------------------------------------------------
