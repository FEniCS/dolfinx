// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <petscpc.h>

#include <dolfin/pcimpl.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/GMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GMRES::GMRES() : LinearSolver(), report(true), ksp(0), B(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Set up solver environment
  KSPCreate(PETSC_COMM_SELF, &ksp);
  KSPSetFromOptions(ksp);  

  //KSPSetType(ksp, KSPGMRES);
  //KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  // Choose preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  //PCSetType(pc, PCNONE);
  //PCSetType(pc, PCILU);
  PCSetFromOptions(pc);

  // Display tolerances
  real _rtol(0.0), _atol(0.0), _dtol(0.0); int _maxiter(0);
  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Setting up PETSc GMRES solver: (rtol, atol, dtol, maxiter) = (%.1e, %.1e, %.1e, %d).",
	      _rtol, _atol, _dtol, _maxiter);
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
void GMRES::solve(const Matrix& A, Vector& x, const Vector& b)
{
  // Check dimensions
  if ( A.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("GMRES solver did not converge.");

  // Report number of iterations
  if ( report )
  {
    //KSPSetMonitor(ksp, KSPDefaultMonitor, PETSC_NULL, PETSC_NULL);
    int its = 0;
    KSPGetIterationNumber(ksp, &its);
    dolfin_info("GMRES converged in %d iterations.", its);
  }
}
//-----------------------------------------------------------------------------
void GMRES::solve(const VirtualMatrix& A, Vector& x, const Vector& b)
{
  // Create preconditioner for virtual system the first time
  //if ( !B )
  // createVirtualPreconditioner(A);

  // Skip preconditioner for now
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCNONE);

  // Check dimensions
  if ( A.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_PRECONDITIONER);
  KSPSolve(ksp, b.vec(), x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("GMRES solver did not converge.");
  
  // Report number of iterations
  if ( report )
  {
    KSPSetMonitor(ksp, KSPDefaultMonitor, PETSC_NULL, PETSC_NULL);
    int its = 0;
    KSPGetIterationNumber(ksp, &its);
    dolfin_info("GMRES converged in %d iterations.", its);
  }
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
  dolfin_info("Changing maxiter for GMRES solver from %e to %e.", _maxiter, maxiter);
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
