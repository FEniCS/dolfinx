// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-02-06
// Last changed: 2006-06-16

#include <time.h>
#include <cstdlib>
#include <cmath>
#include "basic.h"

using namespace dolfin;

namespace dolfin
{
  // Seed only first time
  bool rand_seeded = false;
}

//-----------------------------------------------------------------------------
double dolfin::sqr(double x)
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
double dolfin::rand()
{
  if ( !rand_seeded )
  {
    unsigned int s = static_cast<long int>(time(0));
    std::srand(s);
    rand_seeded = true;
  }
  
  return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}
//-----------------------------------------------------------------------------
void dolfin::seed(unsigned int s)
{
  std::srand(s);
  rand_seeded = true;
}
//-----------------------------------------------------------------------------
