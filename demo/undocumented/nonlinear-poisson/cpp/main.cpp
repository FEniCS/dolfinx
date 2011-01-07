// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2008.
//
// First added:  2005
// Last changed: 2011-01-05
//
// This demo illustrates how to use of DOLFIN for solving a nonlinear
// PDE, in this case a nonlinear variant of Poisson's equation,
//
//     - div (1 + u^2) grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = x*sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = 1  for x = 0
//     du/dn(x, y) = 0  otherwise
//
// This is equivalent to solving
//
//    F(u) = (grad(v), (1 + u^2)*grad(u)) - (v, f) = 0

#include <dolfin.h>
#include "NonlinearPoisson.h"

using namespace dolfin;

// Right-hand side
class Source : public Expression
{
public:

  Source() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0]*sin(x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return std::abs(x[0] - 1.0) < DOLFIN_EPS && on_boundary;
  }
};

int main()
{
  // Use Trilinos if available
  #ifdef HAS_TRILINOS
  parameters["linear_algebra_backend"] = "Epetra";
  #endif

  // Create mesh and define function space
  UnitSquare mesh(16, 16);
  NonlinearPoisson::FunctionSpace V(mesh);

  // Define boundary condition
  DirichletBoundary dirichlet_boundary;
  Constant g(1.0);
  DirichletBC bc(V, g, dirichlet_boundary);

  // Define source and solution functions
  Source f;
  Function u(V);

  // Create (linear) form defining (nonlinear) variational problem
  NonlinearPoisson::LinearForm F(V);
  F.u = u; F.f = f;

  // Create jacobian dF = F' (for use in nonlinear solver).
  NonlinearPoisson::BilinearForm dF(V, V);
  dF.u = u;

  // Solve nonlinear variational problem
  VariationalProblem problem(F, dF, bc);
  problem.solve(u);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
