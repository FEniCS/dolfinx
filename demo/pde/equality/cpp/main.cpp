// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Adapted for EqualityBC demo by Bartosz Sawicki.
//
// First added:  2009-04-15
// Last changed: 2009-10-05
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with homogeneous Dirichlet boundary conditions
// at x = 0, equality boundary conditions at x = 1, and homogeneous
// Neumann conditions on the remaining boundary.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Expression
  {
  public:

    Source() : Expression() {}

    void eval(std::vector<double>& values, const std::vector<double>& x) const
    {
      double dx = x[0] - 0.75;
      double dy = x[1] - 0.5;
      values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS && on_boundary;
    }
  };

  // Sub domain for Equality boundary condition
  class EqualityBoundary : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] > (1.0 - DOLFIN_EPS) && on_boundary;
    }
  };

  // Create mesh
  UnitSquare mesh(32, 32);

  // Create functions
  Source f;

  // Define PDE
  Poisson::FunctionSpace V(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;

  // Create Dirichlet boundary condition
  Constant u0(0.0);
  DirichletBoundary dirichlet_boundary;
  DirichletBC bc0(V, u0, dirichlet_boundary);

  // Create equality boundary condition
  EqualityBoundary equality_boundary;
  EqualityBC bc1(V, equality_boundary);

  // Collect boundary conditions
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc0);
  bcs.push_back(&bc1);

  // Define PDE
  VariationalProblem pde(a, L, bcs);

  // Solve PDE
  Function u(V);
  pde.solve(u);

  // Plot solution
  plot(u);

  return 0;
}
