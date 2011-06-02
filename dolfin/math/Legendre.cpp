// Copyright (C) 2003-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
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
Legendre::Legendre(uint n) : n(n), cache_x(0.0), cache(n + 1)
{
  cache[0] = 1.0; //constant value

  // eval to initialize cache
  eval(n, -1.0);
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
  if (n == 0)
    return 1.0;
  else if (n == 1)
    return x;

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
  // Special cases
  if (n == 0)
    return 0.0;
  else if (n == 1)
    return 1.0;

  // Avoid division by zero
  if (real_abs(x - 1.0) < real_epsilon())
    x -= 2.0*real_epsilon();

  if (real_abs(x + 1.0) < real_epsilon())
    x += 2.0*real_epsilon();

  // Formula, BETA page 254
  const real nn = real(n);
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
  if (real_abs(x - 1.0) < real_epsilon())
    x -= 2.0*real_epsilon();
  if (real_abs(x + 1.0) < real_epsilon())
    x += 2.0*real_epsilon();

  // Formula, BETA page 254
  const real nn = real(n);
  return (2.0*x*ddx(n, x) - nn*(nn+1)*eval(n, x)) / (1.0-x*x);
}
//-----------------------------------------------------------------------------
