// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BASIC_H
#define __BASIC_H

#include <cmath>
#include <dolfin/constants.h>

namespace dolfin
{
  /// Return the square of x
  real sqr(real x);

  /// Return a random number, uniformly distributed between [0.0, 1.0)
  real rand();

  /// Seed random number generator
  void seed(int s);
}

#endif
