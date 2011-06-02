// Copyright (C) 2005 Johan Jansson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2005-2010.
// Modified by Garth N. Wells, 2005-2010.
//
// First added:  2005-12-02
// Last changed: 2011-03-28

#ifdef HAS_PETSC

#include <boost/assign/list_of.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "PETScBaseMatrix.h"
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
                              ("minres",     KSPMINRES)
                              ("tfqmr",      KSPTFQMR)
                              ("richardson", KSPRICHARDSON)
                              ("bicgstab",   KSPBCGS);
//-----------------------------------------------------------------------------
Parameters PETScKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("petsc_krylov_solver");

  p.add("preconditioner_side", "left");

 return p;
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method, std::string pc_type)
  : pc_dolfin(0), preconditioner(new PETScPreconditioner(pc_type)),
    preconditioner_set(false)
{
  // Check that the requested method is known
  if (methods.count(method) == 0)
    error("Requested PETSc Krylov solver '%s' is unknown,", method.c_str());

  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
				                             PETScPreconditioner& preconditioner)
  : preconditioner(reference_to_no_delete_pointer(preconditioner)),
    preconditioner_set(false)

{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
				                             PETScUserPreconditioner& preconditioner)
  : pc_dolfin(&preconditioner), preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(boost::shared_ptr<KSP> _ksp)
  : pc_dolfin(0), _ksp(_ksp), preconditioner_set(true)
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
void PETScKrylovSolver::set_operator(const GenericMatrix& A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operator(const PETScBaseMatrix& A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(const GenericMatrix& A,
                                      const GenericMatrix& P)
{
  this->AA = reference_to_no_delete_pointer(A);
  set_operators(A.down_cast<PETScBaseMatrix>(), P.down_cast<PETScBaseMatrix>());
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(const PETScBaseMatrix& A,
                                      const PETScBaseMatrix& P)
{
  this->A = reference_to_no_delete_pointer(A);
  this->P = reference_to_no_delete_pointer(P);
  assert(this->A);
  assert(this->P);
}
//-----------------------------------------------------------------------------
const GenericMatrix& PETScKrylovSolver::get_operator() const
{
  if (!AA)
    error("Operator for linear solver has not been set.");
  return *AA;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  check_dimensions(get_operator(), x, b);
  return solve(x.down_cast<PETScVector>(), b.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                      const GenericVector& b)
{
  check_dimensions(A, x, b);
  return solve(A.down_cast<PETScBaseMatrix>(), x.down_cast<PETScVector>(),
               b.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(PETScVector& x, const PETScVector& b)
{
  assert(A);
  assert(_ksp);

  // Check dimensions
  const uint N = A->size(1);
  const uint M = A->size(0);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Write a message
  if (parameters["report"] && dolfin::MPI::process_number() == 0)
    log(PROGRESS, "Solving linear system of size %d x %d (PETSc Krylov solver).", M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    x.resize(M);
    x.zero();
  }

  // Set some PETSc-specific options
  set_petsc_options();

  // Set operators
  set_petsc_operators();

  // FIXME: Improve check for re-setting preconditoner, e.g. if parameters change
  // Set preconditioner if necessary
  if (preconditioner && !preconditioner_set)
  {
    preconditioner->set(*this);
    preconditioner_set = true;
  }

  // Solve linear system
  if (MPI::process_number() == 0)
  {
    log(PROGRESS, "PETSc Krylov solver starting to solve %i x %i system.",
        A->size(0), A->size(1));
  }
  KSPSolve(*_ksp, *b.vec(), *x.vec());

  // Get the number of iterations
  int num_iterations = 0;
  KSPGetIterationNumber(*_ksp, &num_iterations);

  // Check if the solution converged and print error/warning if not converged
  KSPConvergedReason reason;
  KSPGetConvergedReason(*_ksp, &reason);
  if (reason < 0)
  {
    // Get solver residual norm
    double rnorm = 0.0;
    KSPGetResidualNorm(*_ksp, &rnorm);
    const char *reason_str = KSPConvergedReasons[reason];
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
      error("PETSc Krylov solver did not converge in %i iterations (PETSc reason %s, norm %e).", num_iterations, reason_str, rnorm);
    else
      warning("Krylov solver did not converge in %i iterations (PETSc reason %s, norm %e).", num_iterations, reason_str, rnorm);
  }

  // Report results
  write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScKrylovSolver::solve(const PETScBaseMatrix& A, PETScVector& x,
                                      const PETScVector& b)
{
  // Check dimensions
  const uint N = A.size(1);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Set operator
  set_operator(A);

  return solve(x, b);
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
void PETScKrylovSolver::init(const std::string& method)
{
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

  // Set solver type
  if (method != "default")
    KSPSetType(*_ksp, methods.find(method)->second);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_operators()
{
  assert(A);
  assert(P);

  // Get some parameters
  const bool reuse_precon = parameters("preconditioner")["reuse"];
  const bool same_pattern = parameters("preconditioner")["same_nonzero_pattern"];

  // Set operators with appropriate option
  if (reuse_precon)
    KSPSetOperators(*_ksp, *A->mat(), *P->mat(), SAME_PRECONDITIONER);
  else if (same_pattern)
    KSPSetOperators(*_ksp, *A->mat(), *P->mat(), SAME_NONZERO_PATTERN);
  else
    KSPSetOperators(*_ksp, *A->mat(), *P->mat(), DIFFERENT_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_options()
{
  // GMRES restart parameter
  KSPGMRESSetRestart(*_ksp, parameters("gmres")["restart"]);

  // Non-zero initial guess
  const bool nonzero_guess = parameters["nonzero_initial_guess"];
  if (nonzero_guess)
    KSPSetInitialGuessNonzero(*_ksp, PETSC_TRUE);
  else
    KSPSetInitialGuessNonzero(*_ksp, PETSC_FALSE);

  if (parameters["monitor_convergence"])
    KSPMonitorSet(*_ksp, KSPMonitorTrueResidualNorm, 0, 0);

  // Set tolerances
  KSPSetTolerances(*_ksp,
                   parameters["relative_tolerance"],
                   parameters["absolute_tolerance"],
                   parameters["divergence_limit"],
                   parameters["maximum_iterations"]);
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

  // FIXME: Get preconditioner description from PETScPreconditioner

  // Report number of iterations and solver type
  if (MPI::process_number() == 0)
  {
    if (reason >= 0)
    {
      log(PROGRESS, "PETSc Krylov solver (%s, %s) converged in %d iterations.",
          ksp_type, pc_type, num_iterations);
    }
    else
    {
      log(PROGRESS, "PETSc Krylov solver (%s, %s) failed to converge in %d iterations.",
          ksp_type, pc_type, num_iterations);
    }

    if (pc_type_str == PCASM || pc_type_str == PCBJACOBI)
    {
      log(PROGRESS, "PETSc Krylov solver preconditioner (%s) sub-methods: (%s, %s)",
          pc_type, sub_ksp_type, sub_pc_type);
    }

    #if PETSC_HAVE_HYPRE
    if (pc_type_str == PCHYPRE)
    {
      const char* hypre_sub_type;
      PCHYPREGetType(pc, &hypre_sub_type);

      log(PROGRESS, "  Hypre preconditioner method: %s", hypre_sub_type);
    }
    #endif
  }
}
//-----------------------------------------------------------------------------

#endif
