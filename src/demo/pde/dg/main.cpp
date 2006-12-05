// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
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

#include "FEM.h"
#include "BilinearForm.h"
#include "LinearForm.h"
#include "Functional.h"
#include <dolfin.h>
#include "DG.h"
#include "EL.h"

  
using namespace dolfin;

int main()
{
  // Right-hand side
  class Source : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 1;
    }
  };

  // Dirichlet boundary condition
  class DirichletBC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      if ( std::abs(p.x() - 0.0) < DOLFIN_EPS )
        value = 0.0;
    }
  };
  
  // Set up problem
  Source f;
  DirichletBC bc;

  UnitSquare mesh(2, 2);
//  Mesh mesh("mesh_dg.xml");

  mesh.init();
  mesh.disp();

  DG::BilinearForm a;
  DG::LinearForm L(f);

  Matrix A;
  Vector b;

  Function U;
  U.init(mesh, a.trial());

  FEM::assemble(a, A, mesh);
  A.disp();

  FEM::assemble(L, b, mesh);
  b.disp();

//cout << mesh.numVertices() << endl;
//cout << mesh.numEdges() << endl;
//cout << mesh.numFacets() << endl;

//  FEM::disp(mesh, a.trial());
  EL el; 
//  FEM::applyBC(A, b, mesh, a.trial(), bc);
  FEM::applyBC(A, b, mesh, el, bc);
//  PDE pde(a, L, mesh, bc);

  // Compute solution
//  Function U = pde.solve();

  LU solver;
  solver.solve(A, U.vector(), b);



  // Save solution to file
  File file("poisson.pvd");
  file << U;

//  File file_mesh("mesh_dg.xml");
//  file_mesh << mesh;

  return 0;
}
