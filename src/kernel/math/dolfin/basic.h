#ifndef __BASIC_H
#define __BASIC_H

#include <cmath>
#include <dolfin/constants.h>

namespace dolfin {

  int max(int x, int y);
  int min(int x, int y);

  /// Return the absolute value of x
  real abs(real x);

  /// Return the square of x
  real sqr(real x);

  /// Return the maximum of x and y
  real max(real x, real y);

  /// Return the minimum of x and y
  real min(real x, real y);

  /// Return a random number, uniformly distributed between [0.0, 1.0)
  real rand();

  int round_int(real x);
  int floor_int(real x);
  int ceil_int(real x);

}

#endif
