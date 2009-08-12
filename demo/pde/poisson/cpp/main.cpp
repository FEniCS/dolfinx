// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2008-12-26
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 500*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// and boundary conditions given by
//
//     u(x, y) = 0 for x = 0 or x = 1

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
    values[0] = 500.0*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  //parameters("linear_algebra_backend") = "uBLAS";

  // Create mesh and function space
  UnitSquare mesh(32, 32);
  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Source f;
  L.f = f;

  // Compute solution
  VariationalProblem problem(a, L, bc);
  problem.parameters("linear_solver") = "iterative";
  Function u;
  problem.solve(u);

  // FIXME: Temporary while testing parallel assembly
  info("Norm of solution vector: %.15g", u.vector().norm());

  // Save solution in VTK format (using base64 encoding)
  File file("poisson.pvd", "base64");
  //File file("poisson.pvd");
  file << u;

  // Plot solution
  plot(u);

  return 0;
}
