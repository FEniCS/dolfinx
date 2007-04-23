// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-05-31
// Last changed: 2006-08-18

#include <dolfin/uBlasILUPreconditioner.h>
#include <dolfin/uBlasDummyPreconditioner.h>
#include <dolfin/uBlasKrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(KrylovMethod method, Preconditioner pc)
  : Parametrized(),
    method(method), pc_user(false), report(false), parameters_read(false)
{
  // Select and create default preconditioner
  selectPreconditioner(pc);
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Preconditioner pc)
  : Parametrized(),
    method(default_method), pc_user(false), report(false), parameters_read(false)
{
  // Select and create default preconditioner
  selectPreconditioner(pc);
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(uBlasPreconditioner& pc)
  : Parametrized(),
    method(default_method), pc(&pc), pc_user(true), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(KrylovMethod method, uBlasPreconditioner& pc)
  : Parametrized(),
    method(method), pc(&pc), pc_user(true), report(false), parameters_read(false)
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
void uBlasKrylovSolver::selectPreconditioner(const Preconditioner preconditioner)
{
  switch(preconditioner)
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
      dolfin_warning("Requested preconditioner is not available for uBlas Krylov solver. Using ILU.");
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
