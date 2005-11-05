// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2004-06-22
// Last changed: 2005-11-24

#include <petscpc.h>

#include <src/ksp/pc/pcimpl.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/GMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GMRES::GMRES() : LinearSolver(), report(true), ksp(0), B(0), M(0), N(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);
}
//-----------------------------------------------------------------------------
GMRES::~GMRES()
{
  // Destroy solver environment.
  if ( ksp ) KSPDestroy(ksp);

  // Destroy matrix B if used
  if ( B ) MatDestroy(B);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const Matrix& A, Vector& x, const Vector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if ( N != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("GMRES solver did not converge.");

  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(ksp, &num_iterations);
  
  // Report number of iterations
  if ( report )
    dolfin_info("GMRES converged in %d iterations.", num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const VirtualMatrix& A, Vector& x, const Vector& b)
{
  // Create preconditioner for virtual system the first time
  //if ( !B )
  // createVirtualPreconditioner(A);

  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if ( N != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  // Skip preconditioner for now
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCNONE);

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("GMRES solver did not converge.");
  
  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(ksp, &num_iterations);
  
  // Report number of iterations
  if ( report )
    dolfin_info("GMRES converged in %d iterations.", num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void GMRES::setReport(bool report)
{
  this->report = report;
}
//-----------------------------------------------------------------------------
void GMRES::setRtol(real rtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);
  
  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing rtol for GMRES solver from %e to %e.", _rtol, rtol);
  _rtol = rtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void GMRES::setAtol(real atol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing atol for GMRES solver from %e to %e.", _atol, atol);
  _atol = atol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void GMRES::setDtol(real dtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing dtol for GMRES solver from %e to %e.", _dtol, dtol);
  _dtol = dtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void GMRES::setMaxiter(int maxiter)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing maxiter for GMRES solver from %d to %d.", _maxiter, maxiter);
  _maxiter = maxiter;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void GMRES::setPreconditioner(Preconditioner &pc)
{
  PC petscpc;
  KSPGetPC(ksp, &petscpc);

  Preconditioner::PCCreate(petscpc);

  petscpc->data = &pc;
  petscpc->ops->apply = Preconditioner::PCApply;
  petscpc->ops->applytranspose = Preconditioner::PCApply;
  petscpc->ops->applysymmetricleft = Preconditioner::PCApply;
  petscpc->ops->applysymmetricright = Preconditioner::PCApply;
}
//-----------------------------------------------------------------------------
void GMRES::disp() const
{
  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
void GMRES::init(uint M, uint N)
{
  // Check if we need to reinitialize
  if ( ksp != 0 && M == this->M && N == this->N )
    return;

  // Don't reinitialize on first solve
  if ( ksp != 0 && this->M == 0 && this->N == 0 )
  {
    this->M = M;
    this->N = N;
    return;
  }

  // Save size of system
  this->M = M;
  this->N = N;

  // Destroy old solver environment if necessary
  if ( ksp != 0 )
    KSPDestroy(ksp);

  // Set up solver environment
  KSPCreate(PETSC_COMM_SELF, &ksp);
  KSPSetFromOptions(ksp);  
  //KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  // Choose preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetFromOptions(pc);

  // Display tolerances
  /*
    real _rtol(0.0), _atol(0.0), _dtol(0.0); int _maxiter(0);
    KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
    dolfin_info("Setting up PETSc GMRES solver: (rtol, atol, dtol, maxiter) = (%.1e, %.1e, %.1e, %d).",
    _rtol, _atol, _dtol, _maxiter);
  */
}
//-----------------------------------------------------------------------------
void GMRES::createVirtualPreconditioner(const VirtualMatrix& A)
{
  // It's probably not a good idea to use a virtual matrix (shell) matrix
  // for the preconditioner matrix

  dolfin_assert(!B);
  MatCreateSeqBAIJ(PETSC_COMM_SELF, 1, A.size(0), A.size(1), 1, PETSC_NULL, &B);
  Vector d(A.size(0));
  d = 1.0;
  MatDiagonalSet(B, d.vec(), INSERT_VALUES);
  //MatView(B, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
