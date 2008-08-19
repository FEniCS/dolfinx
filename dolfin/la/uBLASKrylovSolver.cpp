// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-05-31
// Last changed: 2008-05-15

#include "uBLASILUPreconditioner.h"
#include "uBLASDummyPreconditioner.h"
#include "uBLASKrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(SolverType solver_type, PreconditionerType pc_type)
  : Parametrized(),
    solver_type(solver_type), pc_user(false), report(false), parameters_read(false)
{
  // Select and create default preconditioner
  selectPreconditioner(pc_type);
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(PreconditionerType pc_type)
  : Parametrized(),
    solver_type(default_solver), pc_user(false), report(false), parameters_read(false)
{
  // Select and create default preconditioner
  selectPreconditioner(pc_type);
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(uBLASPreconditioner& pc)
  : Parametrized(),
    solver_type(default_solver), pc(&pc), pc_user(true), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(SolverType solver_type, uBLASPreconditioner& pc)
  : Parametrized(),
    solver_type(solver_type), pc(&pc), pc_user(true), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::~uBLASKrylovSolver()
{
  // Delete preconditioner if it was not created by user
  if( !pc_user )
    delete pc;
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASMatrix<ublas_dense_matrix>& A, 
    uBLASVector& x, const uBLASVector& b)
{ 
  return solveKrylov(A, x, b); 
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASMatrix<ublas_sparse_matrix>& A, 
    uBLASVector& x, const uBLASVector& b)
{ 
  return solveKrylov(A, x, b); 
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASKrylovMatrix& A, uBLASVector& x, 
    const uBLASVector& b)
{ 
  return solveKrylov(A, x, b); 
}
//-----------------------------------------------------------------------------
void uBLASKrylovSolver::selectPreconditioner(PreconditionerType pc_type)
{
  switch (pc_type)
  { 
    case none:
      pc = new uBLASDummyPreconditioner();
      break;
    case ilu:
      pc = new uBLASILUPreconditioner();
      break;
    case default_pc:
      pc = new uBLASILUPreconditioner();
      break;
    default:
      warning("Requested preconditioner is not available for uBLAS Krylov solver. Using ILU.");
      pc = new uBLASILUPreconditioner();
  }
}
//-----------------------------------------------------------------------------
void uBLASKrylovSolver::readParameters()
{
  // Set tolerances and other parameters
  rtol    = get("Krylov relative tolerance");
  atol    = get("Krylov absolute tolerance");
  div_tol = get("Krylov divergence limit");
  max_it  = get("Krylov maximum iterations");
  restart = get("Krylov GMRES restart");
  report  = get("Krylov report");

  // Remember that we have read parameters
  parameters_read = true;
}
//-----------------------------------------------------------------------------
