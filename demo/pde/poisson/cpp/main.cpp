// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2008-10-30
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)
//
// and boundary conditions given by
//
//     u(x, y)     = 0               for x = 0
//     du/dn(x, y) = 25 sin(5 pi y)  for x = 1
//     du/dn(x, y) = 0               otherwise

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term
class Source : public Function
{
  void eval(double* values, const Data& data) const
  {
    double dx = data.x[0] - 0.5;
    double dy = data.x[1] - 0.5;
    values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
  }
};

// Neumann boundary condition
class Flux : public Function
{
  void eval(double* values, const Data& data) const
  {
    double x = data.x[0];
    double y = data.x[1];
    if (x > DOLFIN_EPS)
      values[0] = 25.0*sin(5.0*DOLFIN_PI*y);
    else
      values[0] =  0.0;
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

int main()
{
  // Create mesh and function space
  UnitSquare mesh(32, 32);
  PoissonFunctionSpace V(mesh);

  // Create boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(u0, V, boundary);

  // Create functions
  Source f;
  Flux g;

  // Define PDE
  PoissonBilinearForm a(V, V);
  PoissonLinearForm L(V);
  L.f = f; L.g = g;
  LinearPDE pde(a, L, bc, symmetric);

  // Solve PDE
  Function u;
  pde.solve(u);

  // Plot solution
  plot(u);

  // Save solution to file
  File file("poisson.pvd");
  file << u;

  return 0;
}
