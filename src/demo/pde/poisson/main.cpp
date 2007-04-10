// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-07
// Last changed: 2006-09-05
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

  // Right-hand side
  class Source : public Function
  {
    void eval(real* values, const real* x)
    {
      real dx = x[0] - 0.5;
      real dy = x[1] - 0.5;
      values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }
  };

  // Dirichlet boundary condition
  class DirichletBC : public Function
  {
    void eval(real* values, const real* x)
    {
      values[0] = 0.0;
    }
  };
  
  // Neumann boundary condition
  class NeumannBC : public Function
  {
    void eval(real* values, const real* x)
    {
      if ( std::abs(x[0] - 1.0) < DOLFIN_EPS )
        values[0] = 1.0;
      else
        values[0] = 0.0;
    }
  };

  // Set up problem
  Source f;
  DirichletBC gd;
  NeumannBC gn;


  UnitSquare mesh(3, 3);

  PoissonBilinearForm a;
  PoissonLinearForm L(f, gn);

  Matrix A;
  Vector b;
  assemble(A, a, mesh);
  assemble(b, L, mesh);

  // Define sub domains
  MeshFunction<unsigned int> sub_domains(mesh, 1);
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    sub_domains(*facet) = 0;
    for (VertexIterator vertex(facet); !vertex.end(); ++vertex)
    {
      if ( vertex->x(0) > DOLFIN_EPS )
        sub_domains(*facet) = 1;
    }
  }
  sub_domains.disp();

  // Define boundary condition
  NewBoundaryCondition bc(gd, mesh, sub_domains, 0);

  // Apply boundary condition
  bc.apply(A, b, a);

  cout << "Stiffness matrix " << endl;
  A.disp();
  cout << "RHS vector " << endl;
  b.disp();

/*
  PDE pde(a, L, mesh, bc);

  // Compute solution
  Function U = pde.solve();

  // Save solution to file
  File file("poisson.xml");
  file << U;

*/
  
  return 0;
}
