// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <petsc/petscksp.h>
#include <petsc/petscpc.h>

#include <dolfin/pcimpl.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/VirtualMatrix.h>
#include <dolfin/NewGMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewGMRES::NewGMRES() : report(true), ksp(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Set up solver environment
  dolfin_info("Setting up PETSc solver environment.");
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPGMRES);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  
  // Default: no preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCNONE);
}
//-----------------------------------------------------------------------------
NewGMRES::~NewGMRES()
{
  //Destroy solver environment.
  KSPDestroy(ksp);
}
//-----------------------------------------------------------------------------
void NewGMRES::solve(const NewMatrix& A, NewVector& x, const NewVector& b)
{
  // Check dimensions
  if ( A.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Set linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
  KSPSetRhs(ksp, b.vec());
  KSPSetSolution(ksp, x.vec());

  // Solve linear system
  KSPSolve(ksp);

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
  // Check dimensions
  if ( A.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Set linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
  KSPSetRhs(ksp, b.vec());
  KSPSetSolution(ksp, x.vec());

  // Solve linear system
  KSPSolve(ksp);

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
  _rtol = rtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setAtol(real atol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  _atol = atol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setDtol(real dtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  _dtol = dtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void NewGMRES::setMaxiter(int maxiter)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
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
