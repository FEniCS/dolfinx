// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-07
// Last changed: 2007-04-17
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
  
  // Set up problem
  UnitSquare mesh(3, 3);

  Source f(mesh);
  DirichletBC gd(mesh);
  NeumannBC gn(mesh);

  PoissonBilinearForm a;
  PoissonLinearForm L(f, gn);

  Matrix A;
  Vector b;
  assemble(A, a, mesh);
  assemble(b, L, mesh);

  // Define boundary condition
  DirichletBoundary GD;
  NeumannBoundary GN;
  BoundaryCondition bc(gd, mesh, GD);

  // Apply boundary condition
  bc.apply(A, b, a);

  cout << "Stiffness matrix" << endl;
  A.disp();
  cout << "RHS vector" << endl;
  b.disp();

  Vector x;
  GMRES::solve(A, x, b);

  cout << "Solution vector" << endl;
  x.disp();

  Function u(mesh, x, a);

/*
  PDE pde(a, L, mesh, bc);

  // Compute solution
  Function U = pde.solve();
*/

  // Save solution to file
  File file("poisson.xml");
  file << u;
  
  // Read solution from file
  Function uu;
  file >> uu;

  // Store it to another file for testing
  File file2("poisson2.xml");
  file2 << uu;

  File file3("poisson.pvd");
  file3 << u;

  return 0;
}
