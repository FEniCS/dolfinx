// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2007-08-20
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

int main()
{
  // Source term
  class Source : public Function
  {
  public:    
    double eval(const double* x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      return 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Neumann boundary condition
  class Flux : public Function
  {
  public:
    double eval(const double* x) const
    {
      if (x[0] > DOLFIN_EPS)
        return 25.0*sin(5.0*DOLFIN_PI*x[1]);
      else
        return 0.0;
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

  // Create mesh
  UnitSquare mesh(32, 32);

  // Create functions
  Source f;
  Flux   g;

  // Create boundary condition
  //Function u0(mesh, 0.0);
  //DirichletBoundary boundary;
  //DirichletBC bc(u0, mesh, boundary);
  
  // Define PDE
  PoissonBilinearForm a;
  PoissonLinearForm L(f, g);
  //LinearPDE pde(a, L, mesh, bc, symmetric);
  LinearPDE pde(a, L, mesh, symmetric);

  // Solve PDE
  FunctionSpace V(mesh, a.finite_element(1), a.dofMaps()[1]);
  Function u(V);
  pde.solve(u);

  // Plot solution
  plot(u);

  // Save solution to file
  File file("poisson.pvd");
  file << u;

  return 0;
}
