// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Anders Logg, 2005.

// FIXME: Should not be needed
#include <dolfin/NewFunction.h>
#include <dolfin/BoundaryCondition.h>

// FIXME: Should not be needed
#include <dolfin/NewGMRES.h>

#include <dolfin/Mesh.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewFEM.h>
#include <dolfin/NewBoundaryCondition.h>
#include <dolfin/File.h>
#include <dolfin/Parameter.h>
#include <dolfin.h>

#include "dolfin/Poisson.h"
#include "dolfin/PoissonSolver.h"

// FIXME: Remove when working
#include "dolfin/PoissonOld.h"
#include "dolfin/KrylovSolver.h"
#include "dolfin/FEM.h"

using namespace dolfin;

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh, NewBoundaryCondition& bc)
  : mesh(mesh), bc(bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve()
{
  //cout << "---------------- Old solver -----------------" << endl;
  //solveOld();
  //cout << "---------------- New solver -----------------" << endl;

  Poisson::FiniteElement element;

  // FIXME: Should be able to take f as an argument from main.cpp
  // FIXME: fvalues should be initialized by NewFunction
  NewVector fvalues(mesh.noNodes());
  fvalues = 8.0; // Should together with bc give solution 4*x(1-x)
  NewFunction f(mesh, element, fvalues);

  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  NewMatrix A;
  NewVector x, b;

  // Discretize
  NewFEM::assemble(a, L, A, b, mesh, element);

  // Set boundary conditions
  NewFEM::setBC(A, b, mesh, bc);
  
  //A.disp(false);
  //b.disp();

  // Solve the linear system
  // FIXME: Make NewGMRES::solve() static
  NewGMRES solver;
  solver.solve(A, x, b);

  // FIXME: Remove this and implement output for NewFunction
  Vector xold(b.size());
  for(uint i = 0; i < x.size(); i++)
    xold(i) = x(i);
  Function uold(mesh, xold);
  uold.rename("u", "temperature");
  File file("poisson.m");
  file << uold;
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve(Mesh& mesh, NewBoundaryCondition& bc)
{
  PoissonSolver solver(mesh, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
