// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A library of shape functions

#ifndef __SHAPEFUNCTIONS_H
#define __SHAPEFUNCTIONS_H

#include <dolfin/constants.h>

namespace dolfin {
  
  // Linear shape functions on the reference triangle
  real trilin0(real x, real y, real z, real t);
  real trilin1(real x, real y, real z, real t);
  real trilin2(real x, real y, real z, real t);
  
  // Linear shape functions on the reference tetrahedron
  real tetlin0(real x, real y, real z, real t);
  real tetlin1(real x, real y, real z, real t);
  real tetlin2(real x, real y, real z, real t);
  real tetlin3(real x, real y, real z, real t);

}

#endif
