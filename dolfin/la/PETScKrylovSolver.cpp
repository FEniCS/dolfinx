// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2010.
// Modified by Garth N. Wells, 2005-2010.
//
// First added:  2005-12-02
// Last changed: 2010-05-08

#ifdef HAS_PETSC

#include <boost/assign/list_of.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/main/MPI.h>
#include "KrylovSolver.h"
#include "PETScKrylovMatrix.h"
#include "PETScMatrix.h"
#include "PETScPreconditioner.h"
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
Parameters PETScKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("petsc_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method, std::string pc_type)
  : method(method), pc_dolfin(0),
    preconditioner(new PETScPreconditioner(pc_type)),
    preconditioner_set(false)
{
  // Check that the requested method is known
  if (methods.count(method) == 0)
    error("Requested PETSc Krylov solver '%s' is unknown,", method.c_str());

  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
				                             PETScPreconditioner& preconditioner)
  : method(method),
    preconditioner(reference_to_no_delete_pointer(preconditioner)),
    preconditioner_set(false)

{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
				                             PETScUserPreconditioner& preconditioner)
  : method(method), pc_dolfin(&preconditioner), preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(boost::shared_ptr<KSP> _ksp)
  : method("default"), pc_dolfin(0), _ksp(_ksp),
    preconditioner_set(true)
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
  const uint M = A.size(0);
  const uint N = A.size(1);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Write a message
  if (parameters["report"])
    info(PROGRESS, "Solving linear system of size %d x %d (PETSc Krylov solver).", M, N);

  // Reinitialize KSP solver if necessary
  init(M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    x.resize(M);
    x.zero();
  }

  if (!_ksp)
    _ksp.reset(new KSP, PETScKSPDeleter());

  if (parameters["monitor_convergence"])
    KSPMonitorSet(*_ksp, KSPMonitorTrueResidualNorm, 0, 0);

  // Set tolerances
  KSPSetTolerances(*_ksp,
                   parameters["relative_tolerance"],
                   parameters["absolute_tolerance"],
                   parameters["divergence_limit"],
                   parameters["maximum_iterations"]);

  KSPGMRESSetRestart(*_ksp, parameters["gmres_restart"]);
  KSPSetOperators(*_ksp, *A.mat(), *A.mat(), SAME_NONZERO_PATTERN);

  if (preconditioner && !preconditioner_set)
  {
    preconditioner->set(*this);
    preconditioner_set = true;
  }

  // Solve linear system
  KSPSolve(*_ksp, *b.vec(), *x.vec());

  // Check if the solution converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(*_ksp, &reason);
  if (reason < 0)
  {
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
      error("PETSc Krylov solver did not converge (PETSc reason %i).", reason);
    else
      warning("Krylov solver did not converge (PETSc reason %i).", reason);
  }

  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(*_ksp, &num_iterations);

  // Report results
  write_report(num_iterations, reason);

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
  if (_ksp)
  {
    Mat Amat;
    KSPGetOperators(*_ksp, &Amat, PETSC_NULL, PETSC_NULL);
    uint _M(0), _N(0);
    MatGetSize(Amat, (int*)&_M, (int*)&_N);
    if (M == _M && N == _N)
      return;
  }

  // Check that nobody else shares this solver
  if (_ksp && !_ksp.unique())
    error("Cannot create new KSP Krylov solver. More than one object points to the underlying PETSc object.");

  // Create new KSP object
  _ksp.reset(new KSP, PETScKSPDeleter());

  // Set up solver environment
  if (MPI::num_processes() > 1)
    KSPCreate(PETSC_COMM_WORLD, _ksp.get());
  else
    KSPCreate(PETSC_COMM_SELF, _ksp.get());

  // Set some options
  KSPSetFromOptions(*_ksp);
  KSPSetInitialGuessNonzero(*_ksp, PETSC_TRUE);

  // Set solver type
  if (method != "default")
    KSPSetType(*_ksp, methods.find(method)->second);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::write_report(int num_iterations,
                                     KSPConvergedReason reason)
{
  // Get name of solver and preconditioner
  PC pc;
  const KSPType ksp_type;
  const PCType pc_type;
  KSPGetType(*_ksp, &ksp_type);
  KSPGetPC(*_ksp, &pc);
  PCGetType(pc, &pc_type);

  // If using additive Schwarz or block Jacobi, get 'sub' method which is
  // applied to each block
  const std::string pc_type_str = pc_type;
  const KSPType sub_ksp_type;
  const PCType sub_pc_type;
  PC sub_pc;
  KSP* sub_ksp;
  if (pc_type_str == PCASM || pc_type_str == PCBJACOBI)
  {
    if (pc_type_str == PCASM)
      PCASMGetSubKSP(pc, PETSC_NULL, PETSC_NULL, &sub_ksp);
    else if (pc_type_str == PCBJACOBI)
      PCBJacobiGetSubKSP(pc, PETSC_NULL, PETSC_NULL, &sub_ksp);
    KSPGetType(*sub_ksp, &sub_ksp_type);
    KSPGetPC(*sub_ksp, &sub_pc);
    PCGetType(sub_pc, &sub_pc_type);
  }

  // Report number of iterations and solver type
  if (reason >= 0)
  {
    info(PROGRESS, "PETSc Krylov solver (%s, %s) converged in %d iterations.",
            ksp_type, pc_type, num_iterations);
  }
  else
  {
    info(PROGRESS, "PETSc Krylov solver (%s, %s) failed to converge in %d iterations.",
            ksp_type, pc_type, num_iterations);
  }

  if (pc_type_str == PCASM || pc_type_str == PCBJACOBI)
  {
    info(PROGRESS, "PETSc Krylov solver preconditioner (%s) sub-methods: (%s, %s)",
            pc_type, sub_ksp_type, sub_pc_type);
  }
}
//-----------------------------------------------------------------------------

#endif
