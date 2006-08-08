// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-05-31
// Last changed: 2006-08-08

#include <dolfin/uBlasKrylovSolver.h>

#include <dolfin/uBlasILUPreconditioner.h>
#include <dolfin/uBlasDummyPreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver)
  : Parametrized(),
    type(solver), pc_user(false), report(false), parameters_read(false)
{
  // Create default predefined preconditioner
  pc = new uBlasILUPreconditioner;
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(uBlasPreconditioner::Type preconditioner)
  : Parametrized(),
    type(default_solver), pc_user(false), report(false), parameters_read(false)
{
  // Create predefined preconditioner
  // FIXME: need  to choose appropriate preconditioner here
  pc = new uBlasDummyPreconditioner;
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(uBlasPreconditioner& pc)
  : Parametrized(),
    type(default_solver), pc(&pc), pc_user(true), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver, uBlasPreconditioner::Type preconditioner)
  : Parametrized(),
    type(solver), pc_user(false), report(false), parameters_read(false)
{
  // Create predefined preconditioner
  // FIXME: need  to choose appropriate preconditioner here
  pc = new uBlasDummyPreconditioner;
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver, uBlasPreconditioner& pc)
  : Parametrized(),
    type(type), pc(&pc), pc_user(true), report(false), parameters_read(false)
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
