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

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh) : Solver(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const char* PoissonSolver::description()
{
  return "Poisson's equation";
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve()
{
  Poisson::FiniteElement element;

  // FIXME: Should be able to take f as an argument from main.cpp
  // FIXME: fvalues should be initialized by NewFunction
  NewVector fvalues(mesh.noNodes());
  NewFunction f(mesh, element, fvalues);

  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  NewMatrix A;
  NewVector x, b;

  NewFunction u(mesh, element, x);
  u.rename("u", "temperature");

  // Discretize
  NewFEM::assemble(a, L, A, b, mesh, element);

  //b = 1.0; 
  for (NodeIterator n(mesh); !n.end(); ++n){
    b(n->id()) = 32.0*((1.0-(n->coord().x))*(n->coord().x)+(1.0-(n->coord().y))*(n->coord().y));
  }

  // Set boundary conditions
  dirichletBC(A,b,mesh);
  
  x.init(b.size());
  x = 0.0;

  Matrix Aold(A.size(0), A.size(1));
  Vector bold(b.size());
  Vector xold(x.size());

  for(int i = 0; i < A.size(0); i++)
    for(int j = 0; j < A.size(1); j++)
      Aold(i, j) = A(i, j);

  for(uint i = 0; i < b.size(); i++)
    bold(i) = b(i);

  Aold.show();
  bold.show();

  // Solve the linear system
  // FIXME: Make NewGMRES::solve() static
  NewGMRES solver;
  solver.solve(A, x, b);

  for(uint i = 0; i < x.size(); i++)
    xold(i) = x(i);

  xold.show();

  // Save the solution
  // FIXME: Implement output for NewFunction
  Function uold(mesh, xold, 1);
  uold.rename("u", "temperature");
  File file("poisson.m");
  file << uold;
}
//-----------------------------------------------------------------------------
void PoissonSolver::dirichletBC( NewMatrix& A, NewVector& b, Mesh& mesh)
{
  // Temporary very simple implementation of Dirchlet boundary conditions 

  NewArray<int> bndNodes(mesh.noNodes());
  bndNodes = 0;
  int noBndNodes = 0;
  
  real tol = 1.0e-6;
  for (NodeIterator n(mesh); !n.end(); ++n){
    if ( (fabs(n->coord().x - 0.0)<tol) || 
	 (fabs(n->coord().x - 1.0)<tol) || 
	 (fabs(n->coord().y - 0.0)<tol) || 
	 (fabs(n->coord().y - 1.0)<tol) ){
      bndNodes[n->id()] = 1;
      noBndNodes++;
    }
  }

  int *bndIdx;
  PetscMalloc(noBndNodes*sizeof(int),&bndIdx);
  //NewArray<int> bndIdx(noBndNodes);
  int cnt = 0;
  for (int i=0;i<mesh.noNodes();i++){
    if (bndNodes[i] == 1){
      bndIdx[cnt] = i;
      cnt++;
    }
  }
  
  IS bndRows;
  ISCreateGeneral(PETSC_COMM_SELF, noBndNodes, bndIdx, &bndRows);
  real one = 1.0;
  MatZeroRows( A.mat(), bndRows, &one); CHKERRQ(ierr);
  ISDestroy( bndRows );
  
  for( int i=0; i<noBndNodes;i++)
    b(bndIdx[i]) = 0.0;
}
//-----------------------------------------------------------------------------
