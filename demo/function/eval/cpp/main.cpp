// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-03-11
// Last changed: 2008-11-19
//
// Demonstrating function evaluation at arbitrary points.

#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

#ifdef HAS_GTS

class F : public Function
{
public:
  void eval(double* values, const double* x) const
  {
    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
  }
};

int main()
{
  // Create mesh and a point in the mesh
  UnitCube mesh(8, 8, 8);
  double x[3] = {0.31, 0.32, 0.33};

  // A user-defined function
  F f;

  // Project to a discrete function
  Projection::FunctionSpace V(mesh);
  Projection::BilinearForm a(V, V);
  Projection::LinearForm L(V);
  L.f = f;
  VariationalProblem pde(a, L);
  Function g;
  pde.solve(g);

  // Evaluate user-defined function f
  double value = 0.0;
  f.eval(&value, x);
  info("f(x) = %g", value);

  // Evaluate discrete function g (projection of f)
  g.eval(&value, x);
  info("g(x) = %g", value);
}

#else

int main()
{
  info("DOLFIN must be compiled with GTS to run this demo.");
  return 0;
}

#endif
