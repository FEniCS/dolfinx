// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2006-02-10

#include <dolfin/dolfin_log.h>
#include <dolfin/Function.h>
#include <dolfin/PDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh)
  : Parametrized(), _a(&a), _L(&L), _mesh(&mesh), _bc(0)
{
  // Add parameters
  add("solver", "direct");
}
//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc)
  : Parametrized(), _a(&a), _L(&L), _mesh(&mesh), _bc(&bc)
{
  // Add parameters
  add("solver", "direct");
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function PDE::solve()
{
  Function u;
  return u;
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u)
{
  // Make sure u is a discrete function associated with the trial space
  /*
  Function tmp(
  
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  // Assemble linear system
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);
  
  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);
  
  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson.pvd");
  file << u; 
  */
}
//-----------------------------------------------------------------------------
BilinearForm& PDE::a()
{
  dolfin_assert(_a);
  return *_a;
}
//-----------------------------------------------------------------------------
LinearForm& PDE::L()
{
  dolfin_assert(_L);
  return *_L;
}
//-----------------------------------------------------------------------------
Mesh& PDE::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
