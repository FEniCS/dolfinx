// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A library of shape functions

#ifndef __SHAPEFUNCTIONS_H
#define __SHAPEFUNCTIONS_H

#include <dolfin/constants.h>

namespace dolfin {
  
  // Linear shape functions on the reference triangle
  real tri10(real x, real y, real z, real t);
  real tri11(real x, real y, real z, real t);
  real tri12(real x, real y, real z, real t);
  
  // Linear shape functions on the reference tetrahedron
  real tet10(real x, real y, real z, real t);
  real tet11(real x, real y, real z, real t);
  real tet12(real x, real y, real z, real t);
  real tet13(real x, real y, real z, real t);

}

#endif
