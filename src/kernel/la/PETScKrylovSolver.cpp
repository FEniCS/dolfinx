// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005-2006.
// Modified by Garth N. Wells 2005-2006.
//
// First added:  2005-12-02
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/PETScManager.h>

#if PETSC_VERSION_MAJOR==2 && PETSC_VERSION_MINOR==3 && PETSC_VERSION_SUBMINOR==0
  #include <src/ksp/pc/pcimpl.h>
#else
  #include <private/pcimpl.h>
#endif

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScKrylovSolver.h>

#include <dolfin/PETScSparseMatrix.h>
#include <dolfin/PETScVector.h>
#include <dolfin/VirtualMatrix.h>


using namespace dolfin;

// Monitor function
namespace dolfin
{
  int monitor(KSP ksp, int iteration, real rnorm, void *mctx)
  {
    dolfin_info("Iteration %d: residual = %g", iteration, rnorm);
    return 0;
  }
}

//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver()
  : LinearSolver(),
    type(default_solver), pc_petsc(Preconditioner::default_pc), pc_dolfin(0), 
    ksp(0), M(0), N(0), parameters_read(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(Type solver)
  : LinearSolver(),
    type(solver), pc_petsc(Preconditioner::default_pc), pc_dolfin(0), 
    ksp(0), M(0), N(0), parameters_read(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(Preconditioner::Type preconditioner)
  : LinearSolver(),
    type(default_solver), pc_petsc(preconditioner), pc_dolfin(0),
    ksp(0), M(0), N(0), parameters_read(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(Preconditioner& preconditioner)
  : LinearSolver(),
    type(default_solver), pc_petsc(Preconditioner::default_pc), pc_dolfin(&preconditioner),
    ksp(0), M(0), N(0), parameters_read(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(Type solver, Preconditioner::Type preconditioner)
  : LinearSolver(),
    type(solver), pc_petsc(preconditioner), pc_dolfin(0),
    ksp(0), M(0), N(0), parameters_read(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(Type solver, Preconditioner& preconditioner)
  : LinearSolver(),
    type(solver), pc_petsc(Preconditioner::default_pc), pc_dolfin(&preconditioner),
    ksp(0), M(0), N(0), parameters_read(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::~PETScKrylovSolver()
{
  // Destroy solver environment.
  if ( ksp ) KSPDestroy(ksp);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const PETScSparseMatrix& A, PETScVector& x, const PETScVector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if ( N != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");

  // Write a message
  if ( get("Krylov report") )
    dolfin_info("Solving linear system of size %d x %d (Krylov solver).", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  // Read parameters if not done
  if ( !parameters_read )
    readParameters();

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

  // Report results
  writeReport(num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const VirtualMatrix& A, PETScVector& x, const PETScVector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if ( N != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");
  
  // Write a message
  if ( get("Krylov report") )
    dolfin_info("Solving virtual linear system of size %d x %d (Krylov solver).", M, N);
 
  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  // Read parameters if not done
  if ( !parameters_read )
    readParameters();

  // Don't use preconditioner that can't handle virtual (shell) matrix
  if ( !pc_dolfin )
  {
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
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
  
  // Report results
  writeReport(num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::disp() const
{
  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::init(uint M, uint N)
{
  // Check if we need to reinitialize
  if ( ksp != 0 && M == this->M && N == this->N )
    return;

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

  // Set solver
  setSolver();

  // Set preconditioner
  setPreconditioner();
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::readParameters()
{
  // Don't do anything if not initialized
  if ( !ksp )
    return;

  // Set monitor
  if ( get("Krylov monitor convergence") )
    KSPSetMonitor(ksp, monitor, 0, 0);

  // Set tolerances
  KSPSetTolerances(ksp,
		   get("Krylov relative tolerance"),
		   get("Krylov absolute tolerance"),
		   get("Krylov divergence limit"),
		   get("Krylov maximum iterations"));

  // Set nonzero shift for preconditioner
  if ( !pc_dolfin )
  {
    PC pc;
    KSPGetPC(ksp, &pc);
    PCFactorSetShiftNonzero(pc, get("Krylov shift nonzero"));
  }

  // Remember that we have read parameters
  parameters_read = true;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::setSolver()
{
  // Don't do anything for default method
  if ( type == PETScKrylovSolver::default_solver )
    return;
  
  // Set PETSc Krylov solver
  KSPType ksp_type = getType(type);
  KSPSetType(ksp, ksp_type);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::setPreconditioner()
{
  // Treat special case DOLFIN user-defined preconditioner
  if ( pc_dolfin )
  {
    Preconditioner::setup(ksp, *pc_dolfin);
    return;
  }

  // Treat special case default preconditioner (do nothing)
  if ( pc_petsc == Preconditioner::default_pc )
    return;

  // Get PETSc PC pointer
  PC pc;
  KSPGetPC(ksp, &pc);

  // Treat special case Hypre AMG preconditioner
  if ( pc_petsc == Preconditioner::hypre_amg )
  {  
#if PETSC_HAVE_HYPRE
    PCSetType(pc, PCHYPRE);
    PCHYPRESetType(pc, "boomeramg");
#else
    dolfin_warning("PETSc has not been compiled with the HYPRE library for   "
                   "algerbraic multigrid. Default PETSc solver will be used. "
                   "For performance, installation of HYPRE is recommended.   "
                   "See the DOLFIN user manual for more information.");
#endif
    return;
  }

  // Set preconditioner
  PCSetType(pc, Preconditioner::getType(pc_petsc));
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::writeReport(int num_iterations)
{
  // Check if we should write the report
  bool report = get("Krylov report");
  if ( !report )
    return;
    
  // Get name of solver
  KSPType ksp_type;
  KSPGetType(ksp, &ksp_type);

  // Get name of preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  PCType pc_type;
  PCGetType(pc, &pc_type);

  // Report number of iterations and solver type
  dolfin_info("Krylov solver (%s, %s) converged in %d iterations.",
	      ksp_type, pc_type, num_iterations);
}
//-----------------------------------------------------------------------------
KSPType PETScKrylovSolver::getType(const Type type) const
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
    dolfin_warning("Requested Krylov method unknown. Using GMRES.");
    return KSPGMRES;
  }
}
//-----------------------------------------------------------------------------

#endif
