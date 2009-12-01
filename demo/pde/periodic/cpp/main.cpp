// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-11
// Last changed: 2009-10-05
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with homogeneous Dirichlet boundary conditions
// at y = 0, 1 and periodic boundary conditions at x = 0, 1.

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

    void eval(double* values, const std::vector<double>& x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) + 1.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return (x[1] < DOLFIN_EPS || x[1] > (1.0 - DOLFIN_EPS)) && on_boundary;
    }
  };

  // Sub domain for Periodic boundary condition
  class PeriodicBoundary : public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS && x[0] > -DOLFIN_EPS && on_boundary;
    }

    void map(const double* x, double* y) const
    {
      y[0] = x[0] - 1.0;
      y[1] = x[1];
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

  // Create periodic boundary condition
  PeriodicBoundary periodic_boundary;
  PeriodicBC bc1(V, periodic_boundary);

  // Collect boundary conditions
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc0); bcs.push_back(&bc1);

  // Define PDE
  VariationalProblem pde(a, L, bcs);

  // Solve PDE
  Function u(V);
  pde.solve(u);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("periodic.pvd");
  file << u;

  return 0;
}
