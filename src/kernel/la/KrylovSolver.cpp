// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2005-12-02
// Last changed:

#include <petscpc.h>

#include <src/ksp/pc/pcimpl.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/KrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver() : LinearSolver(), report(true), ksp(0), ksptype(0), 
      pctype(0), B(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(KSPType ksptype) : LinearSolver(), report(true), ksp(0), 
      ksptype(ksptype), pctype(0), B(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(KSPType ksptype, PCType pctype) : LinearSolver(), 
      report(true), ksp(0), ksptype(ksptype), pctype(pctype), B(0), M(0), N(0), 
      dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  // Destroy solver environment.
  if ( ksp ) KSPDestroy(ksp);

  // Destroy matrix B if used
  if ( B ) MatDestroy(B);
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(const Matrix& A, Vector& x, const Vector& b)
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

  // Set Krylov method
  if( ksptype ) KSPSetType(ksp, ksptype);

  PC pc;
  KSPGetPC(ksp, &pc);

  // Override with DOLFIN PC if available, else set the default PETSc PC
  if(dolfin_pc)
  {
    setPreconditioner(*dolfin_pc);
  }
  else
  {
    if(pctype)
    {
      PCSetType(pc, pctype);
    }
    else
    { 
      PCSetFromOptions(pc);
    }
  }
 
  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("Krylov solver did not converge.");

  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(ksp, &num_iterations);
  
  // Report number of iterations
  if ( report )
    dolfin_info("Krylov solver (%s) converged in %d iterations.", ksptype, num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(const VirtualMatrix& A, Vector& x, const Vector& b)
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

  PC pc;
  KSPGetPC(ksp, &pc);
  // PCSetType(pc, PCNONE);

  // Override with DOLFIN PC if available, else set the default PETSc PC
  if(dolfin_pc)
  {
    setPreconditioner(*dolfin_pc);
  }
  else
  {
    PCSetType(pc, PCNONE);
//     PCSetFromOptions(pc);
  }

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  if ( reason < 0 )
    dolfin_error("Krylov solver did not converge.");
  
  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(ksp, &num_iterations);
  
  // Report number of iterations
  if ( report )
    dolfin_info("Krylov converged in %d iterations.", num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setType(KSPType type)
{
  ksptype = type;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setPCType(PCType type)
{
  pctype = type;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setReport(bool report)
{
  this->report = report;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setRtol(real rtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);
  
  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing rtol for Krylov solver from %e to %e.", _rtol, rtol);
  _rtol = rtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void KrylovSolver::setAtol(real atol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing atol for Krylov solver from %e to %e.", _atol, atol);
  _atol = atol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void KrylovSolver::setDtol(real dtol)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing dtol for Krylov solver from %e to %e.", _dtol, dtol);
  _dtol = dtol;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void KrylovSolver::setMaxiter(int maxiter)
{
  real _rtol(0.0), _atol(0.0), _dtol(0.0);
  int _maxiter(0);

  KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
  dolfin_info("Changing maxiter for Krylov solver from %d to %d.", _maxiter, maxiter);
  _maxiter = maxiter;
  KSPSetTolerances(ksp, _rtol, _atol, _dtol, _maxiter);
}
//-----------------------------------------------------------------------------
void KrylovSolver::setPreconditioner(Preconditioner &pc)
{
  // Store pc, to be able to set it again
  dolfin_pc = &pc;

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
void KrylovSolver::disp() const
{
  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
void KrylovSolver::init(uint M, uint N)
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

  // Override with DOLFIN PC if available, else set PETSc PC
  if(dolfin_pc)
  {
    setPreconditioner(*dolfin_pc);
  }
  else
  {
    if(pctype)
    {
      PCSetType(pc, pctype);
    }
    else
    { 
      PCSetFromOptions(pc);
    }
  }

  // Display tolerances
  /*
    real _rtol(0.0), _atol(0.0), _dtol(0.0); int _maxiter(0);
    KSPGetTolerances(ksp, &_rtol, &_atol, &_dtol, &_maxiter);
    dolfin_info("Setting up PETSc Krylov solver: (rtol, atol, dtol, maxiter) = (%.1e, %.1e, %.1e, %d).",
    _rtol, _atol, _dtol, _maxiter);
  */
}
//-----------------------------------------------------------------------------
void KrylovSolver::createVirtualPreconditioner(const VirtualMatrix& A)
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
