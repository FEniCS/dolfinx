// Copyright (C) 2005 Johan Hoffman 
// Licensed under the GNU GPL Version 2.

//#include <dolfin/NSE.h>
#include <dolfin/NSESolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NSESolver::NSESolver(Mesh& mesh, NewFunction& f, 
		     NewBoundaryCondition& bc, NewFunction& u0)
  : mesh(mesh), f(f), bc(bc), u0(u0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NSESolver::solve()
{
  /*
  NewFunction up;
  
  NSEMomentum::FiniteElement element_mom;
  NSEMomentum::BilinearForm a_mom;
  NSEMomentum::LinearForm L_mom(f,u0);

  NSEContinuity::FiniteElement element_con;
  NSEContinuity::BilinearForm a_con;
  NSEContinuity::LinearForm L_con(f,u0);

  NewMatrix A;
  NewVector x, b;

  // Discretize
  NewFEM::assemble(a, L, A, b, mesh, element);

  // Set boundary conditions
  NewFEM::setBC(A, b, mesh, bc);
  
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
  */
}
//-----------------------------------------------------------------------------
void NSESolver::solve(Mesh& mesh, NewFunction& f, 
		      NewBoundaryCondition& bc, NewFunction& u0)
{
  NSESolver solver(mesh, f, bc, u0);
  solver.solve();
}
//-----------------------------------------------------------------------------
