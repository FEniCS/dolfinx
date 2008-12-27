// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2008.
//
// First added:  2005
// Last changed: 2008-12-27
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
class Source : public Function
{
public:
  Source() : t(0) {}

  void eval(double* values, const double* x) const
  {
    values[0] = t*x[0]*sin(x[1]);
  }

  double t;
};

// Dirichlet boundary condition
class DirichletBoundaryCondition : public Function
{
public:
  DirichletBoundaryCondition() : t(0) {}
  
  void eval(double* values, const double* x) const
  {
    values[0] = t;
  }

  double t;
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    return std::abs(x[0] - 1.0) < DOLFIN_EPS && on_boundary;
  }
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Create mesh and function space
  UnitSquare mesh(16, 16);
  NonlinearPoissonFunctionSpace V(mesh);

  // Define boundary condition
  DirichletBoundary dirichlet_boundary;
  DirichletBoundaryCondition g;
  DirichletBC bc(V, g, dirichlet_boundary);

  // Source and solution functions
  Source f;
  Function u;

  // Create forms
  NonlinearPoissonBilinearForm a(V, V);
  a.U = u;
  NonlinearPoissonLinearForm L(V);
  L.U = u; L.f = f;

  // Define nonlinear variational problem
  VariationalProblem problem(a, L, bc, true);
  problem.set("Newton relative tolerance", 1e-6);
  problem.set("Newton convergence criterion", "incremental");

  // Solve variational problem by pseudo time-stepping
  unsigned int n = 3;
  double dt = 1.0 / static_cast<double>(n);
  for (unsigned int i = 0; i < n; i++)
  {
    // Update pseudo-time
    f.t += dt;
    g.t += dt;

    // Solve nonlinear problem
    problem.solve(u);
  }

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
