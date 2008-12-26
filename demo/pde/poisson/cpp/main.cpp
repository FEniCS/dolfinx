// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2008-12-26
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 500*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.02)
//
// and boundary conditions given by
//
//     u(x, y)     = 0               for x = 0,
//     du/dn(x, y) = 25 cos(5 pi y)  for x = 1,
//     du/dn(x, y) = 0               otherwise.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term
class Source : public Function
{
  void eval(double* values, const double* x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
  }
};

// Neumann boundary condition
class Flux : public Function
{
  void eval(double* values, const double* x) const
  {
    if (x[0] > (1.0 - DOLFIN_EPS))
      values[0] = 25.0*cos(5.0*DOLFIN_PI*x[1]);
    else
      values[0] = 0.0;
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh and function space
  UnitSquare mesh(32, 32);
  PoissonFunctionSpace V(mesh);

  // Create boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Create functions
  Source f;
  Flux g;

  // Define PDE
  PoissonBilinearForm a(V, V);
  PoissonLinearForm L(V);
  L.f = f; L.g = g;
  VariationalProblem problem(a, L, bc);

  // Solve PDE
  Function u;
  problem.solve(u);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("poisson.pvd");
  file << u;

  return 0;
}
