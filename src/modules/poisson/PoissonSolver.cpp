// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Anders Logg, 2005.

// FIXME: Should not be needed
#include <dolfin/NewFunction.h>

// FIXME: Should not be needed
#include <dolfin/NewGMRES.h>

#include "Poisson.h"
#include "PoissonSolver.h"

// FIXME: Remove when working
#include "PoissonOld.h"

using namespace dolfin;

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
  cout << "---------------- Old solver -----------------" << endl;

  solveOld();

  cout << "---------------- New solver -----------------" << endl;

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

  //cout << "Before BC:" << endl;
  //A.disp(false);

  // Set boundary conditions
  dirichletBC(A, b, mesh);
  
  //cout << "After BC:" << endl;
  //A.disp(false);

  x.init(b.size());
  x = 0.0;

  // Solve the linear system
  // FIXME: Make NewGMRES::solve() static
  NewGMRES solver;
  solver.solve(A, x, b);

  //A.disp(false);
  //b.disp();

  cout << "New solution x:" << endl;
  x.disp();
    
  Vector xold(b.size());
  for(uint i = 0; i < x.size(); i++)
    xold(i) = x(i);

  cout << "Copied new solution x:" << endl;
  xold.show();

  // Save the solution
  // FIXME: Implement output for NewFunction
  //NewFunction u(mesh, element, x);
  //u.rename("u", "temperature");

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

  //cout << "Old matrix A:" << endl;
  //A.show();

  //cout << "Old vector b:" << endl;
  //b.show();

  // Solve the linear system
  solver.solve(A, x, b);

  cout << "Old solution x:" << endl;
  x.show();

  // Save the solution
  u.rename("u", "temperature");
  file << u;
}
//-----------------------------------------------------------------------------
void PoissonSolver::dirichletBC( NewMatrix& A, NewVector& b, Mesh& mesh)
{
  // Temporary very simple implementation of Dirchlet boundary conditions 

  NewArray<int> bcNodes(mesh.noNodes());
  bcNodes = 0;
  int noBndNodes = 0;
  
  real tol = 1.0e-6;
  for (NodeIterator n(mesh); !n.end(); ++n){
    /*
    if ( (fabs(n->coord().x - 0.0)<tol) || (fabs(n->coord().x - 1.0)<tol) || 
	 (fabs(n->coord().y - 0.0)<tol) || (fabs(n->coord().y - 1.0)<tol) ){
    */
    if ( (fabs(n->coord().x - 0.0)<tol) || (fabs(n->coord().x - 1.0)<tol) ){
      bcNodes[n->id()] = 1;
      noBndNodes++;
    }
  }
  
  NewArray<int> bcIdx(noBndNodes);
  int cnt = 0;
  for (int i=0;i<mesh.noNodes();i++){
    if (bcNodes[i] == 1){
      bcIdx[cnt] = i; 
      cnt++;
    }
  }

  NewArray<real> bcVal(noBndNodes);
  bcVal = 0.0;
  
  NewFEM::setBC(A,b,mesh,bcIdx,bcVal);
}
//-----------------------------------------------------------------------------
