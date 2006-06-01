// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed:

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasKrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver) : type(solver)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
//uBlasKrylovSolver::uBlasKrylovSolver(Preconditioner::Type preconditioner)
//  : type(default_solver), pc_petsc(preconditioner), pc_dolfin(0)
//{
//}
//-----------------------------------------------------------------------------
//uBlasKrylovSolver::uBlasKrylovSolver(Preconditioner& preconditioner) 
//    : type(default_solver), pc_petsc(Preconditioner::default_pc), pc_dolfin(&preconditioner)
//{
//}
//-----------------------------------------------------------------------------
//uBlasKrylovSolver::uBlasKrylovSolver(Type solver, Preconditioner::Type preconditioner)
//  : type(solver), pc_petsc(preconditioner), pc_dolfin(0)
//{
//  // Initialize PETSc
//  PETScManager::init();
//}
//-----------------------------------------------------------------------------
//uBlasKrylovSolver::uBlasKrylovSolver(Type solver, Preconditioner& preconditioner)
//  : type(solver), pc_petsc(Preconditioner::default_pc), pc_dolfin(&preconditioner)
//{
//}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::~uBlasKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
