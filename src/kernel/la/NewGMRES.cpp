// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <petscpc.h>

#include <dolfin/pcimpl.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewGMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewGMRES::NewGMRES() : NewLinearSolver(), report(true), ksp(0), B(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Set up solver environment
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPGMRES);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  
  // Default: no preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  //PCSetType(pc, PCNONE);
  PCSetType(pc, PCILU);
  
  // Display tolerances
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);
  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Setting up PETSc GMRES solver: (rtol, atol, dtol, maxiter) = (%.1e, %.1e, %.1e, %d).",
	      _rtol, _atol, _dtol, _maxiter);

  KSPSetMonitor(ksp, KSPDefaultMonitor, PETSC_NULL, PETSC_NULL);
}
//-----------------------------------------------------------------------------
NewGMRES::~NewGMRES()
{
  // Destroy solver environment.
  if ( ksp ) KSPDestroy(ksp);

  // Destroy matrix B if used
  if ( B ) MatDestroy(B);
}
//-----------------------------------------------------------------------------
void NewGMRES::solve(const NewMatrix& A, NewVector& x, const NewVector& b)
{
  // Check dimensions
  if ( A.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
#ifdef PETSC_2_2_1
  KSPSolve(ksp, b.vec(), x.vec());
#else
  KSPSetRhs(ksp, b.vec());
  KSPSetSolution(ksp, x.vec());
  KSPSolve(ksp);
#endif

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("GMRES solver did not converge.");

  // Report number of iterations
  if ( report )
  {
    int its = 0;
    KSPGetIterationNumber(ksp, &its);
    dolfin_info("GMRES converged in %d iterations.", its);
  }
}
//-----------------------------------------------------------------------------
void NewGMRES::solve(const VirtualMatrix& A, NewVector& x, const NewVector& b)
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

#ifdef PETSC_2_2_1
  KSPSolve(ksp, b.vec(), x.vec());
#else
  KSPSetRhs(ksp, b.vec());
  KSPSetSolution(ksp, x.vec());
  KSPSolve(ksp);
#endif

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("GMRES solver did not converge.");
  
  // Report number of iterations
  if ( report )
  {
    int its = 0;
    KSPGetIterationNumber(ksp, &its);
    dolfin_info("GMRES converged in %d iterations.", its);
  }
}
//-----------------------------------------------------------------------------
void NewGMRES::setReport(bool report)
{
  this->report = report;
}
//-----------------------------------------------------------------------------
void NewGMRES::setRtol(real rtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);
  
  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing rtol for GMRES solver from %e to %e.", _rtol, rtol);
  _rtol = rtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setAtol(real atol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing atol for GMRES solver from %e to %e.", _atol, atol);
  _atol = atol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setDtol(real dtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing dtol for GMRES solver from %e to %e.", _dtol, dtol);
  _dtol = dtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setMaxiter(int maxiter)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing maxiter for GMRES solver from %e to %e.", _maxiter, maxiter);
  _maxiter = maxiter;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setPreconditioner(NewPreconditioner &pc)
{
  PC petscpc;
  KSPGetPC(ksp, &petscpc);

  NewPreconditioner::PCCreate(petscpc);

  petscpc->data = &pc;
  petscpc->ops->apply = NewPreconditioner::PCApply;
  petscpc->ops->applytranspose = NewPreconditioner::PCApply;
  petscpc->ops->applysymmetricleft = NewPreconditioner::PCApply;
  petscpc->ops->applysymmetricright = NewPreconditioner::PCApply;
}
//-----------------------------------------------------------------------------
void NewGMRES::disp() const
{
  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
void NewGMRES::createVirtualPreconditioner(const VirtualMatrix& A)
{
  // It's probably not a good idea to use a virtual matrix (shell) matrix
  // for the preconditioner matrix

  dolfin_assert(!B);
  MatCreateSeqBAIJ(PETSC_COMM_SELF, 1, A.size(0), A.size(1), 1, PETSC_NULL, &B);
  NewVector d(A.size(0));
  d = 1.0;
  MatDiagonalSet(B, d.vec(), INSERT_VALUES);
  //MatView(B, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
