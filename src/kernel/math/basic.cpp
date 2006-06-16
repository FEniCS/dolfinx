// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-02-06
// Last changed: 2006-06-16

#include <time.h>
#include <cstdlib>
#include <cmath>
#include <dolfin/basic.h>

using namespace dolfin;

namespace dolfin
{
  // Seed only first time
  bool rand_seeded = false;
}

//-----------------------------------------------------------------------------
real dolfin::sqr(real x)
{
  return x*x;
}
//-----------------------------------------------------------------------------
dolfin::uint dolfin::ipow(uint a, uint n)
{
  uint p = a;
  for (uint i = 1; i < n; i++)
    p *= a;
  return p;
}
//-----------------------------------------------------------------------------
real dolfin::rand()
{
  if ( !rand_seeded )
  {
    unsigned int s = static_cast<long int>(time(0));
    std::srand(s);
    rand_seeded = true;
  }
  
  return static_cast<real>(std::rand()) / static_cast<real>(RAND_MAX);
}
//-----------------------------------------------------------------------------
void dolfin::seed(unsigned int s)
{
  std::srand(s);
  rand_seeded = true;
}
//-----------------------------------------------------------------------------
