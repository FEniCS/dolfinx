// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005.
//
// First added:  2005
// Last changed: 2007-08-20
//
// This program illustrates the use of DOLFIN for solving a nonlinear
// PDE by solving the nonlinear variant of Poisson's equation,
//
//     - div (1 + u^2) grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = t * x * sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = t  for x = 1
//     du/dn(x, y) = 0  otherwise
//
// where t is pseudo time.
//
// This is equivalent to solving
//
//     F(u) = (grad(v), (1 + u^2)*grad(u)) - f(x, y) = 0

#include <dolfin.h>
#include "NonlinearPoisson.h"
  
using namespace dolfin;

// Right-hand side
class Source : public Function, public TimeDependent
{
public:
  Source(const double* t) : TimeDependent(t) {}
  
  void eval(double* values, const Data& data) const
  {
    double x = data.x[0];
    double y = data.x[1];
    values[0] = time()*x*sin(y);
  }
};

// Dirichlet boundary condition
class DirichletBoundaryCondition : public Function, public TimeDependent
{
public:
  DirichletBoundaryCondition(const double* t) : TimeDependent(t) {}
  
  void eval(double* values, const Data& data) const
  {
    values[0] = 1.0*time();
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

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(16, 16);

  // Pseudo time
  double t = 0.0;

  // Create source function
  Source f(&t);

  // Create function space
  NonlinearPoissonFunctionSpace V(mesh);

  // Solution function
  Function u;

  // Dirichlet boundary conditions
  DirichletBoundary dirichlet_boundary;
  DirichletBoundaryCondition g(&t);
  DirichletBC bc(g, V, dirichlet_boundary);

  // Create forms and PDE
  NonlinearPoissonBilinearForm a(V, V);
  a.U0 = u;
  NonlinearPoissonLinearForm L(V);
  L.U0 = u; L.f = f;
  NonlinearPDE pde(a, L, bc);

  // Solve nonlinear problem in a series of steps
  double dt = 1.0; double T  = 3.0;

  // Solve
  //pde.dolfin_set("Newton relative tolerance", 1e-6); 
  //pde.dolfin_set("Newton convergence criterion", "incremental"); 
  pde.solve(u, t, T, dt);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
