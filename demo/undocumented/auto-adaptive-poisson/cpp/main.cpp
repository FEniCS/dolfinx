// Copyright (C) 2010 Anders Logg and Marie E. Rognes
// Licensed under the GNU LGPL Version 3 or any later version

// First added:  2010-08-19
// Last changed: 2011-01-11

#include <dolfin.h>
#include "AdaptivePoisson.h"

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

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh and function space
  UnitSquare mesh(8, 8);
  AdaptivePoisson::TestSpace V(mesh);    // FIXME: Missing typedefs

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  AdaptivePoisson::BilinearForm a(V, V);
  AdaptivePoisson::LinearForm L(V);
  Source f;
  dUdN g;
  L.f = f;
  L.g = g;
  VariationalProblem pde(a, L, bc);

  // Define function for solution
  Function u(V);

  // Define goal (quantity of interest)
  AdaptivePoisson::GoalFunctional M(mesh);

  // Compute solution (adaptively) with accuracy to within tol
  double tol = 1.e-5;
  pde.solve(u, tol, M);

  summary();

  return 0;
}
