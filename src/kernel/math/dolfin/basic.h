#ifndef __BASIC_H
#define __BASIC_H

#include <dolfin/constants.h>

namespace dolfin {

  int max(int x, int y);
  int min(int x, int y);

  real abs(real x);
  real sqr(real x);
  real max(real x, real y);
  real min(real x, real y);
  real rand();

  int round_int(real x);
  int floor_int(real x);
  int ceil_int(real x);

}

#endif
