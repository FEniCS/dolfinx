// Copyright (C) 2014 Johan Jansson and Garth N. Wells
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
// Modified by Anders Logg 2005-2012
// Modified by Garth N. Wells 2005-2010
// Modified by Fredrik Valdmanis 2011

#ifdef HAS_PETSC

#include <petsclog.h>

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "PETScBaseMatrix.h"
#include "PETScMatrix.h"
#include "PETScPreconditioner.h"
#include "PETScUserPreconditioner.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include "PETScKrylovSolver.h"

using namespace dolfin;

// Map from method string to PETSc (for subset of PETSc solvers)
const std::map<std::string, const KSPType> PETScKrylovSolver::_methods
= { {"default",  ""},
    {"cg",         KSPCG},
    {"gmres",      KSPGMRES},
    {"minres",     KSPMINRES},
    {"tfqmr",      KSPTFQMR},
    {"richardson", KSPRICHARDSON},
    {"bicgstab",   KSPBCGS},
    {"nash",       KSPNASH},
    {"stcg",       KSPSTCG} };

// Map from method string to description
const std::map<std::string, std::string>
PETScKrylovSolver::_methods_descr
=
{ {"default",    "default Krylov method"},
  {"cg",         "Conjugate gradient method"},
  {"gmres",      "Generalized minimal residual method"},
  {"minres",     "Minimal residual method"},
  {"tfqmr",      "Transpose-free quasi-minimal residual method"},
  {"richardson", "Richardson method"},
  {"bicgstab",   "Biconjugate gradient stabilized method"} };

//-----------------------------------------------------------------------------
std::map<std::string, std::string> PETScKrylovSolver::methods()
{
  return PETScKrylovSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> PETScKrylovSolver::preconditioners()
{
  return PETScPreconditioner::preconditioners();
}
//-----------------------------------------------------------------------------
Parameters PETScKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("petsc_krylov_solver");

  // Norm type used in convergence test
  std::set<std::string> allowed_norm_types;
  allowed_norm_types.insert("preconditioned");
  allowed_norm_types.insert("true");
  allowed_norm_types.insert("none");
  p.add("convergence_norm_type", allowed_norm_types);

  return p;
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     std::string preconditioner)
  : _ksp(NULL), pc_dolfin(NULL),
    preconditioner_set(false)
{
   // Check that the requested method is known
  if (_methods.count(method) == 0)
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "create PETSc Krylov solver",
                 "Unknown Krylov method \"%s\"", method.c_str());
  }

  // Set parameter values
  parameters = default_parameters();

  PetscErrorCode ierr;

  // Create PETSc KSP object
  ierr = KSPCreate(PETSC_COMM_WORLD, &_ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");

  // Set Krylov solver type
  if (method != "default")
  {
    ierr = KSPSetType(_ksp, _methods.find(method)->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");
  }

  // Set preconditioner type
  PETScPreconditioner::set_type(*this, preconditioner);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
  std::shared_ptr<PETScPreconditioner> preconditioner)
  : _ksp(NULL), pc_dolfin(NULL), _preconditioner(preconditioner),
  preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  PetscErrorCode ierr;

  // Create PETSc KSP object
  ierr = KSPCreate(PETSC_COMM_WORLD, &_ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");

  // Set Krylov solver type
  if (method != "default")
  {
    ierr = KSPSetType(_ksp, _methods.find(method)->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");
  }
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     std::shared_ptr<PETScUserPreconditioner> preconditioner)
  : _ksp(NULL), pc_dolfin(preconditioner.get()), preconditioner_set(false)
{
  // Set parameter values
  this->parameters = default_parameters();

  PetscErrorCode ierr;

  // Create PETSc KSP object
  ierr = KSPCreate(PETSC_COMM_WORLD, &_ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");

  // Set Krylov solver type
  if (method != "default")
  {
    ierr = KSPSetType(_ksp, _methods.find(method)->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");
  }
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(KSP ksp) : _ksp(ksp), pc_dolfin(0),
                                                preconditioner_set(true)
{
  // Set parameter values
  this->parameters = default_parameters();

  PetscErrorCode ierr;
  if (_ksp)
  {
    // Increment reference count
    ierr = PetscObjectReference((PetscObject)_ksp);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PetscObjectReference");
  }
  else
  {
    // Create PETSc KSP object
    ierr = KSPCreate(PETSC_COMM_WORLD, &_ksp);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");
  }
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::~PETScKrylovSolver()
{
  if (_ksp)
    KSPDestroy(&_ksp);
}
//-----------------------------------------------------------------------------
void
PETScKrylovSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(
  std::shared_ptr<const  GenericLinearOperator> A,
  std::shared_ptr<const GenericLinearOperator> P)
{
  _set_operators(as_type<const PETScBaseMatrix>(A),
                 as_type<const PETScBaseMatrix>(P));
}
//-----------------------------------------------------------------------------
const PETScBaseMatrix& PETScKrylovSolver::get_operator() const
{
  if (!_matA)
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "access operator for PETSc Krylov solver",
                 "Operator has not been set");
  }
  return *_matA;
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  return solve(as_type<PETScVector>(x), as_type<const PETScVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(const GenericLinearOperator& A,
                                     GenericVector& x,
                                     const GenericVector& b)
{
  return _solve(as_type<const PETScBaseMatrix>(A),
                as_type<PETScVector>(x),
                as_type<const PETScVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(PETScVector& x, const PETScVector& b)
{
  Timer timer("PETSc Krylov solver");

  dolfin_assert(_matA);
  dolfin_assert(_ksp);

  PetscErrorCode ierr;

  // Check dimensions
  const std::size_t M = _matA->size(0);
  const std::size_t N = _matA->size(1);
  if (_matA->size(0) != b.size())
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "unable to solve linear system with PETSc Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %ld rows and right-hand side vector has %ld rows)",
                 _matA->size(0), b.size());
  }

  // Write a message
  const bool report = this->parameters["report"];
  if (report && dolfin::MPI::rank(this->mpi_comm()) == 0)
  {
    info("Solving linear system of size %ld x %ld (PETSc Krylov solver).",
         M, N);
  }

  // Initialize solution vector, if necessary
  if (x.empty())
  {
    _matA->init_vector(x, 1);
    x.zero();
  }

  // Set some PETSc-specific options
  set_petsc_ksp_options();

  // FIXME: Improve check for re-setting preconditioner, e.g. if
  //        parameters change
  // FIXME: Solve using matrix free matrices fails if no user provided
  //        Prec is provided
  // Set preconditioner if necessary
  if (_preconditioner && !preconditioner_set)
  {
    _preconditioner->set(*this);
    preconditioner_set = true;
  }
  else if (pc_dolfin && !preconditioner_set)
  {
    // User defined preconditioner
    PETScUserPreconditioner::setup(_ksp, *pc_dolfin);
    preconditioner_set = true;
  }

  // Set convergence norm type
  if (this->parameters["convergence_norm_type"].is_set())
  {
    const std::string convergence_norm_type
      = this->parameters["convergence_norm_type"];
    if (convergence_norm_type == "true")
    {
      ierr = KSPSetNormType(_ksp, KSP_NORM_UNPRECONDITIONED);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetNormType");
    }
    else if (convergence_norm_type == "preconditioned")
    {
      ierr = KSPSetNormType(_ksp, KSP_NORM_PRECONDITIONED);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetNormType");
    }
    else if (convergence_norm_type == "none")
    {
      ierr = KSPSetNormType(_ksp, KSP_NORM_NONE);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetNormType");
    }
  }

  // Solve linear system
  if (dolfin::MPI::rank(this->mpi_comm()) == 0)
  {
    log(PROGRESS, "PETSc Krylov solver starting to solve %i x %i system.",
        _matA->size(0), _matA->size(1));
  }

  ierr =  KSPSolve(_ksp, b.vec(), x.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSolve");

  // Update ghost values
  x.update_ghost_values();

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = KSPGetIterationNumber(_ksp, &num_iterations);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetIterationNumber");

  // Check if the solution converged and print error/warning if not
  // converged
  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(_ksp, &reason);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetConvergedReason");
  if (reason < 0)
  {
    // Get solver residual norm
    double rnorm = 0.0;
    ierr = KSPGetResidualNorm(_ksp, &rnorm);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetResidualNorm");
    const char *reason_str = KSPConvergedReasons[reason];
    bool error_on_nonconvergence = this->parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
    {
      dolfin_error("PETScKrylovSolver.cpp",
                   "solve linear system using PETSc Krylov solver",
                   "Solution failed to converge in %i iterations (PETSc reason %s, residual norm ||r|| = %e)",
                   static_cast<int>(num_iterations), reason_str, rnorm);
    }
    else
    {
      warning("Krylov solver did not converge in %i iterations (PETSc reason %s, residual norm ||r|| = %e).",
              num_iterations, reason_str, rnorm);
    }
  }

  // Report results
  if (report && dolfin::MPI::rank(this->mpi_comm()) == 0)
    write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_reuse_preconditioner(bool reuse_pc)
{
  dolfin_assert(_ksp);
  const PetscBool _reuse_pc = reuse_pc ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode ierr = KSPSetReusePreconditioner(_ksp, _reuse_pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetReusePreconditioner");
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix (if any)
  dolfin_assert(_ksp);
  PetscErrorCode ierr = KSPSetOptionsPrefix(_ksp, options_prefix.c_str());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScKrylovSolver::get_options_prefix() const
{
  dolfin_assert(_ksp);
  const char* prefix = NULL;
  PetscErrorCode ierr = KSPGetOptionsPrefix(_ksp, &prefix);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
std::string PETScKrylovSolver::str(bool verbose) const
{
  dolfin_assert(_ksp);
  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for PETScKrylovSolver not implemented, calling \
PETSc KSPView directly.");
    PetscErrorCode ierr = KSPView(_ksp, PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPView");
  }
  else
    s << "<PETScKrylovSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
MPI_Comm PETScKrylovSolver::mpi_comm() const
{
  dolfin_assert(_ksp);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_ksp, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
KSP PETScKrylovSolver::ksp() const
{
  return _ksp;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::_set_operator(std::shared_ptr<const PETScBaseMatrix> A)
{
  _set_operators(A, A);
}
//-----------------------------------------------------------------------------
void
PETScKrylovSolver::_set_operators(std::shared_ptr<const PETScBaseMatrix> A,
                                  std::shared_ptr<const PETScBaseMatrix> P)
{
  _matA = A;
  _matP = P;
  dolfin_assert(_matA);
  dolfin_assert(_matP);
  dolfin_assert(_ksp);

  dolfin_assert(_ksp);
  PetscErrorCode ierr;
  ierr = KSPSetOperators(_ksp, _matA->mat(), _matP->mat());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::_solve(const PETScBaseMatrix& A, PETScVector& x,
                                      const PETScVector& b)
{
  // Set operator
  std::shared_ptr<const PETScBaseMatrix> Atmp(&A, NoDeleter());
  _set_operator(Atmp);

  // Call solve
  return solve(x, b);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_ksp_options()
{
  dolfin_assert(_ksp);
  PetscErrorCode ierr;

  // GMRES restart parameter
  const int gmres_restart = this->parameters("gmres")["restart"];
  ierr = KSPGMRESSetRestart(_ksp, gmres_restart);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGMRESSetRestart");

  // Non-zero initial guess
  const bool nonzero_guess = this->parameters["nonzero_initial_guess"];
  PetscBool petsc_nonzero_guess = PETSC_FALSE;
  if (nonzero_guess)
    petsc_nonzero_guess = PETSC_TRUE;
  ierr = KSPSetInitialGuessNonzero(_ksp, petsc_nonzero_guess);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetInitialGuessNonzero");

  // Monitor convergence
  const bool monitor_convergence = this->parameters["monitor_convergence"];
  if (monitor_convergence)
  {
    ierr = KSPMonitorSet(_ksp, KSPMonitorTrueResidualNorm,
                         PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)_ksp)),
                         NULL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPMonitorSet");
  }

  // Set tolerances
  const int max_iterations = this->parameters["maximum_iterations"];
  ierr = KSPSetTolerances(_ksp,
                          this->parameters["relative_tolerance"],
                          this->parameters["absolute_tolerance"],
                          this->parameters["divergence_limit"],
                          max_iterations);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetTolerances");
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::write_report(int num_iterations,
                                     KSPConvergedReason reason)
{
  dolfin_assert(_ksp);

  PetscErrorCode ierr;

  // Get name of solver and preconditioner
  PC pc;
  KSPType ksp_type;
  PCType pc_type;

  ierr = KSPGetType(_ksp, &ksp_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetType");

  ierr = KSPGetPC(_ksp, &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  ierr = PCGetType(pc, &pc_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCGetType");

  // If using additive Schwarz or block Jacobi, get 'sub' method which
  // is applied to each block
  const std::string pc_type_str = pc_type;
  KSPType sub_ksp_type;
  PCType sub_pc_type;
  PC sub_pc;
  KSP* sub_ksp = NULL;
  if (pc_type_str == PCASM || pc_type_str == PCBJACOBI)
  {
    if (pc_type_str == PCASM)
    {
      ierr = PCASMGetSubKSP(pc, NULL, NULL, &sub_ksp);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCASMGetSubKSP");
    }
    else if (pc_type_str == PCBJACOBI)
    {
      ierr = PCBJacobiGetSubKSP(pc, NULL, NULL, &sub_ksp);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCBJacobiGetSubKSP");
    }
    ierr = KSPGetType(*sub_ksp, &sub_ksp_type);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetType");

    ierr = KSPGetPC(*sub_ksp, &sub_pc);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

    ierr = PCGetType(sub_pc, &sub_pc_type);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCGetType");
  }

  // FIXME: Get preconditioner description from PETScPreconditioner

  // Report number of iterations and solver type
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
    log(PROGRESS, "PETSc Krylov solver preconditioner (%s) submethods: (%s, %s)",
        pc_type, sub_ksp_type, sub_pc_type);
  }

  #if PETSC_HAVE_HYPRE
  if (pc_type_str == PCHYPRE)
  {
    const char* hypre_sub_type;
    ierr = PCHYPREGetType(pc, &hypre_sub_type);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCHYPREGetType");

    log(PROGRESS, "  Hypre preconditioner method: %s", hypre_sub_type);
  }
  #endif
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::check_dimensions(const PETScBaseMatrix& A,
                                         const GenericVector& x,
                                         const GenericVector& b) const
{
  // Check dimensions of A
  if (A.size(0) == 0 || A.size(1) == 0)
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "unable to solve linear system with PETSc Krylov solver",
                 "Matrix does not have a nonzero number of rows and columns");
  }

  // Check dimensions of A vs b
  if (A.size(0) != b.size())
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "unable to solve linear system with PETSc Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %ld rows and right-hand side vector has %ld rows)",
                 A.size(0), b.size());
  }

  // Check dimensions of A vs x
  if (!x.empty() && x.size() != A.size(1))
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "unable to solve linear system with PETSc Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %ld columns and solution vector has %ld rows)",
                 A.size(1), x.size());
  }
}
//-----------------------------------------------------------------------------

#endif
