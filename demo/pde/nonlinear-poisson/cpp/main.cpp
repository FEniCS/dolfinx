// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005
//
// First added:  2005
// Last changed: 2007-08-20
//
// This program illustrates the use of the DOLFIN for solving a nonlinear PDE
// by solving the nonlinear variant of Poisson's equation
//
//     - div (1+u^2) grad u(x, y) = f(x, y)
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
// This is equivalent to solving: 
// F(u) = (grad(v), (1-u^2)*grad(u)) - f(x,y) = 0

#include <dolfin.h>
#include "NonlinearPoisson.h"
  
using namespace dolfin;

// Right-hand side
class Source : public Function, public TimeDependent
{
  public:

    Source(Mesh& mesh, const real* t) : Function(mesh), TimeDependent(t) {}

    real eval(const real* x) const
    {
      return time()*x[0]*sin(x[1]);
    }
};

// Dirichlet boundary condition
class DirichletBoundaryCondition : public Function, public TimeDependent
{
  public:
    DirichletBoundaryCondition(Mesh& mesh, const real* t) : Function(mesh), TimeDependent(t) {}

    real eval(const real* x) const
    {
      return 1.0*time();
    }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
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
  real t = 0.0;

  // Create source function
  Source f(mesh, &t);

  // Dirichlet boundary conditions
  DirichletBoundary dirichlet_boundary;
  DirichletBoundaryCondition g(mesh, &t);
  DirichletBC bc(g, mesh, dirichlet_boundary);

  // Solution function
  Function u;

  // Create forms and nonlinear PDE
  NonlinearPoissonBilinearForm a(u);
  NonlinearPoissonLinearForm L(u, f);
  NonlinearPDE pde(a, L, mesh, bc);

  // Solve nonlinear problem in a series of steps
  real dt = 1.0; real T  = 3.0;

//  pde.dolfin_set("Newton relative tolerance", 1e-6); 
//  pde.dolfin_set("Newton convergence criterion", "incremental"); 

  // Solve
  pde.solve(u, t, T, dt);

  // Plot solution
  plot(u);

  // Save function to file
  File file("nonlinear_poisson.pvd");
  file << u;

  return 0;
}
