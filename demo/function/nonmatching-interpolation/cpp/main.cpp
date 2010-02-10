// Copyright (C) 2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-06-17
// Last changed:

//
// This program demonstrates the interpolation of functions on non-matching
// meshes.
//

#include <dolfin.h>
#include "P1.h"
#include "P3.h"

using namespace dolfin;

#ifdef HAS_CGAL

class MyExpression : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(10.0*x[0])*sin(10.0*x[1]);
  }

};

int main()
{
  // Create meshes
  UnitSquare mesh0(16, 16);
  UnitSquare mesh1(64, 64);

  // Create function spaces
  P3::FunctionSpace V0(mesh0);
  P1::FunctionSpace V1(mesh1);

  // Create functions
  Function f0(V0);
  Function f1(V1);

  // Interpolate expression into V0
  MyExpression e;
  f0.interpolate(e);

  // Interpolate V0 function (coarse mesh) into V1 function space (fine mesh)
  f1.interpolate(f0);

  // Plot results
  plot(f0);
  plot(f1);

  return 0;
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
