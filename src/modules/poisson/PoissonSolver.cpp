// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "PoissonSolver.h"
#include "FFCPoisson.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh) : Solver(mesh)
{
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
  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  //  FFCPoissonLinearForm L(element);
  NewMatrix A;
  NewVector x,b;

  b.init(A.size(0));
  b = 1.0;

  //  NewFunction  u(mesh, x);
  //  NewFunction  f("source");
  //  NewGMRES     solver;
  File         file("poisson.m");

  // Discretise
  //NewFEM::assemble(a, L, mesh, A, b);
  NewFEM::assemble(a, mesh, A);

  // Solve the linear system
  //solver.solve(A, x, b);

  // Save the solution
  //u.rename("u", "temperature");
  //file << u;
}
//-----------------------------------------------------------------------------
