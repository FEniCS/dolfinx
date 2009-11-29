// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2008.
//
// First added:  2005
// Last changed: 2009-10-05
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

  void eval(std::vector<double>& values, const std::vector<double>& x) const
  {
    values[0] = x[0]*sin(x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return std::abs(x[0] - 1.0) < DOLFIN_EPS && on_boundary;
  }
};

int main()
{
  // Create mesh and define function space
  UnitSquare mesh(4, 4);
  NonlinearPoisson::FunctionSpace V(mesh);

  // Define boundary condition
  DirichletBoundary dirichlet_boundary;
  Constant g(1.0);
  DirichletBC bc(V, g, dirichlet_boundary);

  // Define source and solution functions
  Source f;
  Function u(V);

  // Create forms
  NonlinearPoisson::BilinearForm a(V, V);
  a.u = u;
  NonlinearPoisson::LinearForm L(V);
  L.u = u; L.f = f;

  // Solve nonlinear variational problem
  VariationalProblem problem(a, L, bc, true);
  problem.solve(u);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
