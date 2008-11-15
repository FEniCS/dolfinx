// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-03-11
// Last changed: 2008-11-14
//
// Demonstrating function evaluation at arbitrary points.

#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

class F : public Function
{
public:
  
  void eval(double* values, const Data& data) const
  {
    double x = data.x[0];
    double y = data.x[1];
    double z = data.x[2];
    values[0] =  sin(3.0*x)*sin(3.0*y)*sin(3.0*z);
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
  ProjectionFunctionSpace V(mesh);
  ProjectionBilinearForm a(V, V);
  ProjectionLinearForm L(V);
  L.f = f;
  LinearPDE pde(a, L);
  Function g;
  pde.solve(g);

  // Prepare dat for function evaluation
  Data data;
  data.x = x;
  double value;

  // Evaluate user-defined function f
  f.eval(&value, data);
  message("f(x) = %g", value);

  // Evaluate discrete function g (projection of f)
  g.eval(&value, data);
  message("g(x) = %g", value);
}
