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
  NewVector fvalues;
  NewFunction f(mesh, element, fvalues);

  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  NewMatrix A;
  NewVector x, b;

  NewFunction u(mesh, element, x);
  u.rename("u", "temperature");

  // Discretize
  NewFEM::assemble(a, L, A, b, mesh, element);

  // Solve the linear system
  // FIXME: Make NewGMRES::solve() static
  NewGMRES solver;
  solver.solve(A, x, b);

  // Save the solution
  // FIXME: Implement output for NewFunction
  //File file("poisson.m");
  //file << u;
}
//-----------------------------------------------------------------------------
