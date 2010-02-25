// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2009.
// Modified by Garth N. Wells, 2005-2009.
//
// First added:  2005-12-02
// Last changed: 2010-02-16

#ifdef HAS_PETSC

#include <boost/assign/list_of.hpp>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/main/MPI.h>
#include "KrylovSolver.h"
#include "PETScKrylovMatrix.h"
#include "PETScMatrix.h"
#include "PETScUserPreconditioner.h"
#include "PETScVector.h"
#include "PETScKrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
namespace dolfin
{
  class PETScKSPDeleter
  {
  public:
    void operator() (KSP* _ksp)
    {
      if (_ksp)
        KSPDestroy(*_ksp);
      delete _ksp;
    }
  };
}
//-----------------------------------------------------------------------------
// Available solvers
const std::map<std::string, const KSPType> PETScKrylovSolver::methods
  = boost::assign::map_list_of("default",  "")
                              ("cg",         KSPCG)
                              ("gmres",      KSPGMRES)
                              ("richardson", KSPRICHARDSON)
                              ("bicgstab",   KSPBCGS);
//-----------------------------------------------------------------------------
// Available preconditioners
const std::map<std::string, const PCType> PETScKrylovSolver::pc_methods
  = boost::assign::map_list_of("default",   "")
                              ("none",      PCNONE)
                              ("ilu",       PCILU)
                              ("jacobi",    PCJACOBI)
                              ("sor",       PCSOR)
                              ("icc",       PCICC)
                              ("amg_hypre", PCHYPRE)
                              ("amg_ml",    PCML);
//-----------------------------------------------------------------------------
Parameters PETScKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("petsc_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method, std::string pc_type)
  : method(method), pc_petsc(pc_type), pc_dolfin(0),
    _ksp(static_cast<KSP*>(0), PETScKSPDeleter()), M(0), N(0),
    parameters_read(false), pc_set(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
				     PETScUserPreconditioner& preconditioner)
  : method(method), pc_petsc("default"), pc_dolfin(&preconditioner),
    _ksp(static_cast<KSP*>(0), PETScKSPDeleter()), M(0), N(0),
    parameters_read(false), pc_set(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(boost::shared_ptr<KSP> _ksp)
  : method("default"), pc_petsc("default"), pc_dolfin(0),
    _ksp(_ksp), M(0), N(0), parameters_read(false), pc_set(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------

PETScKrylovSolver::~PETScKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                       const GenericVector& b)
{
  return solve(A.down_cast<PETScMatrix>(), x.down_cast<PETScVector>(),
               b.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const PETScMatrix& A, PETScVector& x,
                                      const PETScVector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Write a message
  if (parameters["report"])
    info("Solving linear system of size %d x %d (Krylov solver).", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    x.resize(M);
    x.zero();
  }

  // Read parameters if not done
  if (!parameters_read)
    read_parameters();

  if (!_ksp)
    _ksp.reset(new KSP, PETScKSPDeleter());

  KSPGMRESSetRestart(*_ksp, parameters["gmres_restart"]);
  KSPSetOperators(*_ksp, *A.mat(), *A.mat(), SAME_NONZERO_PATTERN);

  // FIXME: Preconditioner being set here and not in init() to avoid PETSc bug
  //        with Hypre. See explanation inside PETScKrylovSolver:init().
  if (!pc_set)
  {
    set_petsc_preconditioner();
    pc_set = true;
  }

  // Solve linear system
  KSPSolve(*_ksp, *b.vec(), *x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(*_ksp, &reason);
  if (reason < 0)
    warning("Krylov solver did not converge.");

  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(*_ksp, &num_iterations);

  // Report results
  write_report(num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const PETScKrylovMatrix& A,
                                      PETScVector& x, const PETScVector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Write a message
  if (parameters["report"])
    info("Solving virtual linear system of size %d x %d (Krylov solver).", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    x.resize(M);
    x.zero();
  }

  // Read parameters if not done
  if (!parameters_read)
    read_parameters();

  // Don't use preconditioner that can't handle virtual (shell) matrix
  if (!pc_dolfin)
  {
    PC pc;
    KSPGetPC(*_ksp, &pc);
    PCSetType(pc, PCNONE);
  }

  // Solve linear system
  KSPSetOperators(*_ksp, A.mat(), A.mat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(*_ksp, *b.vec(), *x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(*_ksp, &reason);
  if (reason < 0)
    error("Krylov solver did not converge.");

  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(*_ksp, &num_iterations);

  // Report results
  write_report(num_iterations);

  return num_iterations;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<KSP> PETScKrylovSolver::ksp() const
{
  return _ksp;
}
//-----------------------------------------------------------------------------
std::string PETScKrylovSolver::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for PETScKrylovSolver not implemented, calling PETSc KSPView directly.");
    KSPView(*_ksp, PETSC_VIEWER_STDOUT_WORLD);
  }
  else
    s << "<PETScKrylovSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::init(uint M, uint N)
{
  // Check if we need to reinitialize
  if (_ksp != 0 && M == this->M && N == this->N)
    return;

  // Save size of system
  this->M = M;
  this->N = N;

  // Destroy old solver environment if necessary
  if (!_ksp.unique())
    error("Cannot create new KSP Krylov solver. More than one object points to the underlying PETSc object.");

  _ksp.reset(new KSP, PETScKSPDeleter());

  // Set up solver environment
  if (MPI::num_processes() > 1)
  {
    info("Creating parallel PETSc Krylov solver.");
    KSPCreate(PETSC_COMM_WORLD, _ksp.get());
  }
  else
    KSPCreate(PETSC_COMM_SELF, _ksp.get());

  // Set some options
  KSPSetFromOptions(*_ksp);
  KSPSetInitialGuessNonzero(*_ksp, PETSC_TRUE);

  // Check that the requested method is known
  if (methods.count(method) == 0)
    error("Requested PETSc Krylov solver '%s' is unknown,", method.c_str());

  // Set solver
  if (method != "default")
    KSPSetType(*_ksp, methods.find(method)->second);

  //set_solver();

  // FIXME: The preconditioner is being set in solve() due to a PETSc bug
  //        when using Hypre preconditioner. The problem can be avoided by
  //        setting the preconditioner after KSPSetOperators(). This will be
  //        fixed in PETSc and then the preconditioner can be set here again.
  // Set preconditioner
  //  set_petsc_preconditioner();
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::read_parameters()
{
  // Don't do anything if not initialized
  if (!_ksp)
    return;

  // Set monitor
  if (parameters["monitor_convergence"])
    KSPMonitorSet(*_ksp, KSPMonitorTrueResidualNorm, 0, 0);

  // Set tolerances
  KSPSetTolerances(*_ksp,
		   parameters["relative_tolerance"],
		   parameters["absolute_tolerance"],
		   parameters["divergence_limit"],
		   parameters["maximum_iterations"]);

  // Set nonzero shift for preconditioner
  if (!pc_dolfin)
  {
    PC pc;
    KSPGetPC(*_ksp, &pc);
    PCFactorSetShiftNonzero(pc, parameters["shift_nonzero"]);
  }

  // Remember that we have read parameters
  parameters_read = true;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_preconditioner()
{
  // Treat special case DOLFIN user-defined preconditioner
  if (pc_dolfin)
  {
    PETScUserPreconditioner::setup(*_ksp, *pc_dolfin);
    return;
  }

  // Treat special case default preconditioner (do nothing)
  if (pc_petsc == "default")
    return;

  // Check that the requested method is known
  if (pc_methods.count(pc_petsc) == 0)
    error("Requested PETSc proconditioner '%s' is unknown,", pc_petsc.c_str());

  // Get PETSc PC pointer
  PC pc;
  KSPGetPC(*_ksp, &pc);

  // Make sure options are set
  PCSetFromOptions(pc);

  // Treat special case Hypre AMG preconditioner
  if (pc_petsc == "amg_hypre")
  {
#if PETSC_HAVE_HYPRE
    PCSetType(pc, PCHYPRE);
    PCHYPRESetType(pc, "boomeramg");
    PCSetFromOptions(pc);
#else
    warning("PETSc has not been compiled with the HYPRE library for   "
                   "algebraic multigrid. Default PETSc solver will be used. "
                   "For performance, installation of HYPRE is recommended.   "
                   "See the DOLFIN user manual for more information.");
#endif
    return;
  }

  // Treat special case ML AMG preconditioner
  if (pc_petsc == "amg_ml")
  {
#if PETSC_HAVE_ML
  PCSetType(pc, pc_methods.find(pc_petsc)->second);
  PCFactorSetShiftNonzero(pc, PETSC_DECIDE);
#else
    warning("PETSc has not been compiled with the ML library for   "
                   "algerbraic multigrid. Default PETSc solver will be used. "
                   "For performance, installation of ML is recommended.");
#endif
    return;
  }

  // Set preconditioner
  PCSetType(pc, pc_methods.find(pc_petsc)->second);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::write_report(int num_iterations)
{
  // Check if we should write the report
  if (!parameters["report"])
    return;

  // Get name of solver and preconditioner
  const KSPType _ksp_type;
  const PCType pc_type;

  PC pc;
  KSPGetType(*_ksp, &_ksp_type);
  KSPGetPC(*_ksp, &pc);
  PCGetType(pc, &pc_type);

  // Report number of iterations and solver type
  info("PETSc Krylov solver (%s, %s) converged in %d iterations.",
          _ksp_type, pc_type, num_iterations);
}
//-----------------------------------------------------------------------------

#endif
