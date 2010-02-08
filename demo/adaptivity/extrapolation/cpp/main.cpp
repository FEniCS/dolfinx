// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-08
// Last changed: 2010-02-08
//
// This program demonstrates extrapolation of a P1 function to a P2
// function on the same mesh. This is useful for postprocessing a
// computed dual approximation for use in an error estimate.

#include <dolfin.h>
#include "P1.h"
#include "P2.h"

using namespace dolfin;

class Dual : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5.0*x[0])*sin(5.0*x[1]);
  }

};

int main()
{
  // Create mesh and function spaces
  UnitSquare mesh(8, 8);
  P1::FunctionSpace P1(mesh);
  P2::FunctionSpace P2(mesh);

  // Create exact dual
  Dual dual;

  // Create P1 approximation of exact dual
  Function z1(P1);
  z1.interpolate(dual);

  // Create P2 approximation from P1 approximation
  Function z2(P2);
  z2.extrapolate(z1);

  // Plot approximations
  plot(z1, "z1");
  plot(z2, "z2");

  return 0;
}
