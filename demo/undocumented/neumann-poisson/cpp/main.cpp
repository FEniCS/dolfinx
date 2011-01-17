// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2011-01-17
//
// This demo program illustrates how to solve Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with pure Neumann boundary conditions:
//
//     du/dn(x, y) = -sin(5*x)
//
// and source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// Since only Neumann conditions are applied, u is only determined up to
// a constant c by the above equations. An addition constraint is thus
// required, for instance
//
//   \int u = 0
//
// This can be accomplished by introducing the constant c as an
// additional unknown (to be sought in the space of real numbers)
// and the above constraint.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Boundary flux (Neumann boundary condition)
class Flux : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -sin(5*x[0]);
  }
};

int main()
{

  not_working_in_parallel("neumann-poisson demo (with space of reals)");

  // Create mesh and function space
  UnitSquare mesh(64, 64);
  Poisson::FunctionSpace V(mesh);

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Source f;
  Flux g;
  L.f = f;
  L.g = g;

  // Compute solution
  VariationalProblem problem(a, L);
  problem.parameters["solver"]["linear_solver"] = "iterative";
  Function w(V);
  problem.solve(w);

  Function u = w[0];

  // Plot solution
  plot(u);

  return 0;
}
