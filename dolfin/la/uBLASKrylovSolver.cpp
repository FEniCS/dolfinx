// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-05-31
// Last changed: 2008-05-15

#include "uBlasILUPreconditioner.h"
#include "uBlasDummyPreconditioner.h"
#include "uBlasKrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(SolverType solver_type, PreconditionerType pc_type)
  : Parametrized(),
    solver_type(solver_type), pc_user(false), report(false), parameters_read(false)
{
  // Select and create default preconditioner
  selectPreconditioner(pc_type);
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(PreconditionerType pc_type)
  : Parametrized(),
    solver_type(default_solver), pc_user(false), report(false), parameters_read(false)
{
  // Select and create default preconditioner
  selectPreconditioner(pc_type);
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(uBlasPreconditioner& pc)
  : Parametrized(),
    solver_type(default_solver), pc(&pc), pc_user(true), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(SolverType solver_type, uBlasPreconditioner& pc)
  : Parametrized(),
    solver_type(solver_type), pc(&pc), pc_user(true), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::~uBlasKrylovSolver()
{
  // Delete preconditioner if it was not created by user
  if( !pc_user )
    delete pc;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::solve(const uBlasMatrix<ublas_dense_matrix>& A, 
    uBlasVector& x, const uBlasVector& b)
{ 
  return solveKrylov(A, x, b); 
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, 
    uBlasVector& x, const uBlasVector& b)
{ 
  return solveKrylov(A, x, b); 
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::solve(const uBlasKrylovMatrix& A, uBlasVector& x, 
    const uBlasVector& b)
{ 
  return solveKrylov(A, x, b); 
}
//-----------------------------------------------------------------------------
void uBlasKrylovSolver::selectPreconditioner(PreconditionerType pc_type)
{
  switch (pc_type)
  { 
    case none:
      pc = new uBlasDummyPreconditioner();
      break;
    case ilu:
      pc = new uBlasILUPreconditioner();
      break;
    case default_pc:
      pc = new uBlasILUPreconditioner();
      break;
    default:
      warning("Requested preconditioner is not available for uBlas Krylov solver. Using ILU.");
      pc = new uBlasILUPreconditioner();
  }
}
//-----------------------------------------------------------------------------
void uBlasKrylovSolver::readParameters()
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
