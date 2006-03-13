// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005-2006.
// Modified by Garth N. Wells 2005.
//
// First added:  2005-12-02
// Last changed: 2006-03-13

#include <petscpc.h>

#if PETSC_VERSION_MAJOR==2 && PETSC_VERSION_MINOR==3 && PETSC_VERSION_SUBMINOR==0
  #include <src/ksp/pc/pcimpl.h>
#else
  #include <private/pcimpl.h>
#endif

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/KrylovSolver.h>

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
KrylovSolver::KrylovSolver()
  : LinearSolver(),
    set_pc(true),
    solver_type(default_solver),
    preconditioner_type(Preconditioner::default_pc), 
    ksp(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Read parameters
  readParameters();
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Type solver_type)
  : LinearSolver(),
    set_pc(true),
    solver_type(solver_type),
    preconditioner_type(Preconditioner::default_pc), 
    ksp(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Read parameters
  readParameters();
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Preconditioner::Type preconditioner_type)
  : LinearSolver(), 
    set_pc(true),
    solver_type(default_solver),
    preconditioner_type(preconditioner_type), 
    ksp(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Read parameters
  readParameters();
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Type solver_type, Preconditioner::Type preconditioner_type)
  : LinearSolver(),
    set_pc(true),
    solver_type(solver_type), 
    preconditioner_type(preconditioner_type),
    ksp(0), M(0), N(0), dolfin_pc(0)
{
  // Initialize PETSc
  PETScManager::init();
  
  // Initialize KSP solver
  init(0, 0);

  // Read parameters
  readParameters();
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  // Destroy solver environment.
  if ( ksp ) KSPDestroy(ksp);
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
  if ( get("Krylov report") )
    dolfin_info("Solving linear system of size %d x %d (Krylov solver).", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  x.init(M);

  // Read parameters
  readParameters();

  // Set Krylov method and preconditioner
  KSPType ksp_type;
  ksp_type = getType(solver_type);

  if (solver_type != KrylovSolver::default_solver) 
    KSPSetType(ksp, ksp_type);

  PC pc;
  KSPGetPC(ksp, &pc);

  // Use DOLFIN PC if available, else set PETSc PC
  if (dolfin_pc)
  {
    setPreconditioner(*dolfin_pc);
  }
  else
  {
    setPreconditioner(pc);
    //    if(preconditioner_type != Preconditioner::default_pc) 
    //      PCSetType(pc, pc_type);
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
  
  // Get solver and preconditioner type for output
  PCType pc_type;
  KSPGetType(ksp, &ksp_type);
  PCGetType(pc, &pc_type);

  // Report number of iterations and solver type
  if ( get("Krylov report") )
    dolfin_info("Krylov solver (%s, %s) converged in %d iterations.", ksp_type, pc_type, num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(const VirtualMatrix& A, Vector& x, const Vector& b)
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

  // Read parameters
  readParameters();

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
  if ( get("Krylov report") )
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
void KrylovSolver::setPreconditioner(Preconditioner &pc)
{
  // Store pc, to be able to set it again
  dolfin_pc = &pc;

  // Setup DOLFIN preconditioner
  Preconditioner::setup(ksp, pc);
}
//-----------------------------------------------------------------------------
KSP KrylovSolver::solver()
{
  dolfin_assert(ksp);
  return ksp;
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
void KrylovSolver::readParameters()
{
  // Set monitor
  if ( get("Krylov monitor convergence") )
    KSPSetMonitor(ksp, monitor, 0, 0);
  
  // Set tolerances
  KSPSetTolerances(ksp,
		   get("Krylov relative tolerance"),
		   get("Krylov absolute tolerance"),
		   get("Krylov divergence limit"),
		   get("Krylov maximum iterations"));
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

