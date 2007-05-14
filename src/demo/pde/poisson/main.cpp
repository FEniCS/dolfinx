// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2007-05-02
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
//     u(x, y)     = 0  for x = 0
//     du/dn(x, y) = 1  for x = 1
//     du/dn(x, y) = 0  otherwise

#include <dolfin.h>
#include "Poisson.h"
  
using namespace dolfin;

int main()
{
  // Source term
  class Source : public Function
  {
  public:
    
    Source(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x)
    {
      real dx = x[0] - 0.5;
      real dy = x[1] - 0.5;
      return 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Dirichlet boundary condition
  class DirichletBC : public Function
  {
  public:

    DirichletBC(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x)
    {
      return 0.0;
    }

  };

  // FIXME: Use sub domain, not condition in function
  
  // Neumann boundary condition
  class NeumannBC : public Function
  {
  public:

    NeumannBC(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x)
    {
      if ( std::abs(x[0] - 1.0) < DOLFIN_EPS )
        return 1.0;
      else
        return 0.0;
    }

  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] < DOLFIN_EPS && on_boundary;
    }
  };

  // Sub domain for Neumann boundary condition
  class NeumannBoundary : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] > 1.0 - DOLFIN_EPS && on_boundary;
    }
  };

  // Create mesh
  UnitSquare mesh(16, 16);

  // Create functions
  Source f(mesh);
  DirichletBC gd(mesh);
  NeumannBC gn(mesh);

  // Create sub domains
  DirichletBoundary GD;
  NeumannBoundary GN;
  
  // Define PDE
  PoissonBilinearForm a;
  PoissonLinearForm L(f, gn);
  BoundaryCondition bc(gd, mesh, GD);
  PDE pde(a, L, mesh, bc);

  // Solve PDE
  Function u;
  pde.solve(u);

  // Plot solution
  plot(u);

  // Save solution to file
  File file("poisson.xml");
  file << u;

  return 0;
}
