// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

real phi0(real x, real y, real z, real t)
{
  return 1 - x - y;
}

real phi1(real x, real y, real z, real t)
{
  return x;
}

real phi2(real x, real y, real z, real t)
{
  return y;
}

void main()
{
  // Definition of shape functions
  FunctionSpace::ShapeFunction v0(phi0);
  FunctionSpace::ShapeFunction v1(phi1);
  FunctionSpace::ShapeFunction v2(phi2);
  
  // Some shape function algebra
  FunctionSpace::ElementFunction v = v1 * (v0 + v1 * v2);
  cout << "v(0.1, 0.2, 0.3, 0.0) = " << v(0.1, 0.2, 0.3, 0.0) << endl;
  
}
