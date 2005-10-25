// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#include <dolfin/PETScManager.h>
#include <dolfin/NonlinearSolver.h>

using namespace dolfin;
//-----------------------------------------------------------------------------
NonlinearSolver::NonlinearSolver() : snes(0), M(0), N(0)
{
  // Initialize PETSc
  PETScManager::init();

  // Initialize SNES solver
  init(0, 0);
}
//-----------------------------------------------------------------------------
NonlinearSolver::~NonlinearSolver()
{
  if( snes ) SNESDestroy(snes); 
}
//-----------------------------------------------------------------------------
void NonlinearSolver::solve(Vector& x)
{
//FIXME
}
//-----------------------------------------------------------------------------
void NonlinearSolver::init(uint M, uint N)
{
//FIXME
}
//-----------------------------------------------------------------------------
// NonlinearFunctional
//-----------------------------------------------------------------------------
NonlinearFunctional::NonlinearFunctional() : mesh(0), a(0), L(0), b(0), x(0)
{
//FIXME
}
//-----------------------------------------------------------------------------
NonlinearFunctional::~NonlinearFunctional()
{
// Do nothing 
}
//-----------------------------------------------------------------------------
