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

#include "Poisson.h"
#include "PoissonSolver.h"

// FIXME: Remove when working
#include "PoissonOld.h"

using namespace dolfin;

class MyBC : public NewBoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( (fabs(p.x - 0.0) < DOLFIN_EPS) || (fabs(p.x - 1.0) < DOLFIN_EPS ) )
      value.set(0.0);
    
    return value;
  }
};

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh) : Solver(mesh)
{
  // FIXME: Remove when working
  dolfin_parameter(Parameter::FUNCTION, "source", 0);
}
//-----------------------------------------------------------------------------
const char* PoissonSolver::description()
{
  return "Poisson's equation";
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve()
{
  //cout << "---------------- Old solver -----------------" << endl;
  //solveOld();
  //cout << "---------------- New solver -----------------" << endl;

  // FIXME: This should be input to the solver
  MyBC bc;

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
void PoissonSolver::solveOld()
{
  // This is for comparison with the old solver, remove when working

  Matrix       A;
  Vector       x, b;
  Function     u(mesh, x);
  Function     f("source");
  PoissonOld   poisson(f);
  KrylovSolver solver;
  File         file("poissonold.m");

  // Discretise
  FEM::assemble(poisson, mesh, A, b);

  cout << "Old matrix A:" << endl;
  A.show();

  cout << "Old vector b:" << endl;
  b.show();

  // Solve the linear system
  solver.solve(A, x, b);

  cout << "Old solution x:" << endl;
  x.show();

  // Save the solution
  u.rename("u", "temperature");
  file << u;
}
//-----------------------------------------------------------------------------
