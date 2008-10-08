// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-03-11
// Last changed: 2007-03-17
//
// Testing evaluation at arbitrary points

#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

class F : public Function
{
public:
  
  F(Mesh& mesh) : Function(mesh) {}

  double eval(const double* x) const
  {
    return sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
  }
  
};

int main()
{
  // Create mesh and a point in the mesh
  UnitCube mesh(8, 8, 8);
  double x[3] = {0.3, 0.3, 0.3};

  // A user-defined function
  F f(mesh);

  // Project to a discrete function
  ProjectionBilinearForm a;
  ProjectionLinearForm L(f);
  LinearPDE pde(a, L, mesh);
  Function g;
  pde.solve(g);

  // Evaluate user-defined function f
  message("f(x) = %g", f.eval(x));

  // Evaluate discrete function g (projection of f)
  message("g(x) = %g", g.eval(x));
}
