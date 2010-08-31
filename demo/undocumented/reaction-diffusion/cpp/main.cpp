// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-06-15
// Last changed: 2010-01-27
//
// This demo program solves the reaction-diffusion equation
//
//    - div grad u + u = f
//
// on the unit square with f = sin(x)*sin(y) and homogeneous Neumann
// boundary conditions.

#include <dolfin.h>
#include "ReactionDiffusion.h"

using namespace dolfin;

// Source term
class Source : public Expression
{
public:

  Source() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(x[0])*sin(x[1]);
  }

};

int main()
{
  // Define variational problem
  UnitSquare mesh(32, 32);
  Source f;
  ReactionDiffusion::FunctionSpace V(mesh);
  ReactionDiffusion::BilinearForm a(V, V);
  ReactionDiffusion::LinearForm L(V, f);

  // Compute and plot solution
  VariationalProblem problem(a, L);
  Function u(V);
  problem.solve(u);
  plot(u);

  return 0;
}
