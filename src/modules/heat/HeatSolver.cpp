// Copyright (C) 2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.

//#include <dolfin/Heat.h>
#include <dolfin/HeatSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HeatSolver::HeatSolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc, 
		       NewFunction& u0, real dt, real T0, real T) 
  : mesh(mesh), f(f), bc(bc), u0(u0), dt(dt), T0(T0), T(T) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void HeatSolver::solve()
{
  /*
  Heat::FiniteElement element;

  Heat::BilinearForm a;
  Heat::LinearForm L(f);

  NewMatrix A;
  NewVector x, xprev, b;
  
  NewGMRES solver;

  real t = T0;
  
  while (t<T)
  {
    // Update previous solution 
    xprev = x;

    // Discretize
    NewFEM::assemble(a, L, A, b, mesh, element);

    // Set boundary conditions
    NewFEM::setBC(A, b, mesh, bc);
  
    // Solve the linear system
    solver.solve(A, x, b);

  }
  */

  /*
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
void HeatSolver::solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc, 
		       NewFunction& u0, real dt, real T0, real T) 
{
  HeatSolver solver(mesh, f, bc, u0, dt, T0, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
