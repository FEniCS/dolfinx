// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005.
// Modified by Anders Logg, 2005.

#include <time.h>
#include <stdlib.h>
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
real dolfin::rand()
{
  if ( !rand_seeded ) {
    long int seed = static_cast<long int>(time(0));
    srand48(seed);
    rand_seeded = true;
  }
  
  return static_cast<real>(drand48());
}
//-----------------------------------------------------------------------------
void dolfin::seed(int s)
{
  srand48(static_cast<long int>(s));
  rand_seeded = true;
}
//-----------------------------------------------------------------------------
