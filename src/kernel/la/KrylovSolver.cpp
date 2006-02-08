// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005, 2006.
// Modified by Garth N. Wells 2005.
//
// First added:  2005-12-02
// Last changed: 2006-02-08

#include <petscpc.h>

#include <src/ksp/pc/pcimpl.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/KrylovSolver.h>

using namespace dolfin;

// Monitor function
namespace dolfin
{
  int monitor(KSP ksp, int iteration, real rnorm, void *mctx)
  {
    dolfin_info("residual = %g", rnorm);
    return 0;
  }
}

//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver() : LinearSolver(), set_pc(true), report(true), 
      solver_type(default_solver), preconditioner_type(Preconditioner::default_pc), 
      ksp(0), B(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Add parameters
  add("monitor convergence", false);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Type solver_type) : LinearSolver(), set_pc(true), report(true), 
       solver_type(solver_type), preconditioner_type(Preconditioner::default_pc), 
       ksp(0), B(0), M(0), N(0), dolfin_pc(0)

{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Add parameters
  add("monitor convergence", false);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Preconditioner::Type preconditioner_type) : LinearSolver(), 
       set_pc(true), report(true), solver_type(default_solver), preconditioner_type(preconditioner_type), 
       ksp(0), B(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Add parameters
  add("monitor convergence", false);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Type solver_type, Preconditioner::Type preconditioner_type) : 
      LinearSolver(), set_pc(true), report(true), solver_type(solver_type), 
      preconditioner_type(preconditioner_type), ksp(0), B(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Add parameters
  add("monitor convergence", false);
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

  // Write a message
  if ( report )
    dolfin_info("Solving linear system of size %d x %d.", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  // Set Krylov method and preconditioner
  KSPType ksp_type;
  ksp_type = getType(solver_type);

  if(solver_type != KrylovSolver::default_solver) 
    KSPSetType(ksp, ksp_type);

  PC pc;
  KSPGetPC(ksp, &pc);

  // Use DOLFIN PC if available, else set PETSc PC
  if(dolfin_pc)
  {
    setPreconditioner(*dolfin_pc);
  }
  else
  {
    setPreconditioner(pc);
//    if(preconditioner_type != Preconditioner::default_pc) 
//      PCSetType(pc, pc_type);
  }

  // Set monitor
  if ( get("monitor convergence") )
    KSPSetMonitor(ksp, monitor, 0, 0);
 
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
  
  // Get solver and preconditioner type for output
  PCType pc_type;
  KSPGetType(ksp, &ksp_type);
  PCGetType(pc, &pc_type);

  // Report number of iterations and solver type
  if ( report )
    dolfin_info("Krylov solver (%s, %s) converged in %d iterations.", ksp_type, pc_type, num_iterations);

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

  // Write a message
  if ( report )
    dolfin_info("Solving linear system of size %d x %d.", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  PC pc;
  KSPGetPC(ksp, &pc);
  // PCSetType(pc, PCNONE);

  // Use DOLFIN PC if available, else set the default PETSc PC
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
void KrylovSolver::setType(const Type type)
{
  solver_type = type;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setPreconditioner(const Preconditioner::Type type)
{
  if(type != preconditioner_type)
    set_pc = true; 

  preconditioner_type = type;
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

  // Setup DOLFIN preconditioner
  Preconditioner::setup(ksp, pc);
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
KSPType KrylovSolver::getType(const Type type) const
{
  switch (type)
  {
  case bicgstab:
    return KSPBCGS;
  case cg:
    return KSPCG;
  case default_solver:
    return "default";
  case gmres:
    return KSPGMRES;
  default:
    dolfin_warning("Requested Krylov method unkown. Using GMRES.");
    return KSPGMRES;
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::setPreconditioner(PC &pc)
{
  PCType pc_type;
  pc_type = Preconditioner::getType(preconditioner_type);

  if(preconditioner_type != Preconditioner::default_pc)
  {
    if(preconditioner_type == Preconditioner::hypre_amg) 
    {  
      setPreconditionerHypre(pc);
    }
    else
    {
      PCSetType(pc, pc_type);
    }
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::setPreconditionerHypre(PC &pc)
{
  // Check that PETSc was compiled with HYPRE
  #ifdef PETSC_HAVE_HYPRE
    if(set_pc)
    {
      PCSetType(pc, PCHYPRE);
      PCHYPRESetType(pc, "boomeramg");
    } 
    set_pc = false; 
 #else
    dolfin_warning("PETSc has not been compiled with the HYPRE library for   "
                   "algerbraic multigrid. Default PETSc solver will be used. "
                   "For performance, installation of HYPRE is recommended.   "
                   "See the DOLFIN user manual. ");
  #endif
}
//-----------------------------------------------------------------------------

