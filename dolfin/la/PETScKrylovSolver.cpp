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
// Modified by Anders Logg 2005-2012
// Modified by Garth N. Wells 2005-2010
// Modified by Fredrik Valdmanis 2011
//
// First added:  2005-12-02
// Last changed: 2013-11-25

#ifdef HAS_PETSC

#include <petsclog.h>
#include <boost/assign/list_of.hpp>

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

// Mapping from method string to PETSc
const std::map<std::string, const KSPType> PETScKrylovSolver::_methods
  = boost::assign::map_list_of("default",  "")
                              ("cg",         KSPCG)
                              ("gmres",      KSPGMRES)
                              ("minres",     KSPMINRES)
                              ("tfqmr",      KSPTFQMR)
                              ("richardson", KSPRICHARDSON)
                              ("bicgstab",   KSPBCGS);

// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
  PETScKrylovSolver::_methods_descr = boost::assign::pair_list_of
    ("default",    "default Krylov method")
    ("cg",         "Conjugate gradient method")
    ("gmres",      "Generalized minimal residual method")
    ("minres",     "Minimal residual method")
    ("tfqmr",      "Transpose-free quasi-minimal residual method")
    ("richardson", "Richardson method")
    ("bicgstab",   "Biconjugate gradient stabilized method");

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
PETScKrylovSolver::methods()
{
  return PETScKrylovSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
PETScKrylovSolver::preconditioners()
{
  return PETScPreconditioner::preconditioners();
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_options_prefix(std::string prefix)
{
  dolfin_assert(_ksp);
  PetscErrorCode ierr = KSPSetOptionsPrefix(_ksp, prefix.c_str());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
Parameters PETScKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("petsc_krylov_solver");

  // Norm type used in covergence test
  std::set<std::string> allowed_norm_types;
  allowed_norm_types.insert("preconditioned");
  allowed_norm_types.insert("true");
  allowed_norm_types.insert("none");
  p.add("convergence_norm_type", allowed_norm_types);

  // Control PETSc performance profiling
  p.add<bool>("profile");

  p.add("options_prefix", "default");

  return p;
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     std::string preconditioner)
  : _ksp(NULL), pc_dolfin(NULL),
    _preconditioner(new PETScPreconditioner(preconditioner)),
    petsc_nullspace(NULL), preconditioner_set(false)
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

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     PETScPreconditioner& preconditioner)
  : _ksp(NULL), _preconditioner(reference_to_no_delete_pointer(preconditioner)),
    petsc_nullspace(NULL), preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
  std::shared_ptr<PETScPreconditioner> preconditioner)
  : _ksp(NULL), _preconditioner(preconditioner), petsc_nullspace(NULL),
    preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     PETScUserPreconditioner& preconditioner)
  : _ksp(NULL), pc_dolfin(&preconditioner), petsc_nullspace(NULL),
    preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
  std::shared_ptr<PETScUserPreconditioner> preconditioner)
  : _ksp(NULL), pc_dolfin(preconditioner.get()), petsc_nullspace(NULL),
    preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(KSP ksp)
  : _ksp(ksp), pc_dolfin(0), petsc_nullspace(NULL), preconditioner_set(true)
{
  // Set parameter values
  parameters = default_parameters();

  // Increment reference count
  if (_ksp)
    PetscObjectReference((PetscObject)_ksp);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::~PETScKrylovSolver()
{
  if (_ksp)
    KSPDestroy(&_ksp);
  if (petsc_nullspace)
    MatNullSpaceDestroy(&petsc_nullspace);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operator(std::shared_ptr<const PETScBaseMatrix> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(std::shared_ptr<const  GenericLinearOperator> A,
                                      std::shared_ptr<const GenericLinearOperator> P)
{
  set_operators(as_type<const PETScBaseMatrix>(A),
                as_type<const PETScBaseMatrix>(P));
}
//-----------------------------------------------------------------------------
void
PETScKrylovSolver::set_operators(std::shared_ptr<const PETScBaseMatrix> A,
                                 std::shared_ptr<const PETScBaseMatrix> P)
{
  _matA = A;
  _matP = P;
  dolfin_assert(_matA);
  dolfin_assert(_matP);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_nullspace(const VectorSpaceBasis& nullspace)
{
  PetscErrorCode ierr;

  // Copy vectors
  for (std::size_t i = 0; i < nullspace.dim(); ++i)
  {
    dolfin_assert(nullspace[i]);
    const PETScVector& x = nullspace[i]->down_cast<PETScVector>();

    // Copy vector
    _nullspace.push_back(x);
  }

  // Get pointers to underlying PETSc objects and normalize vectors
  std::vector<Vec> petsc_vec(nullspace.dim());
  for (std::size_t i = 0; i < nullspace.dim(); ++i)
  {
    petsc_vec[i] = _nullspace[i].vec();
    PetscReal val = 0.0;
    ierr = VecNormalize(_nullspace[i].vec(), &val);
    if (ierr != 0) petsc_error(ierr, __FILE__, "VecNormalize");
  }

  // Create null space
  if (petsc_nullspace)
    MatNullSpaceDestroy(&petsc_nullspace);
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE,
                            nullspace.dim(),
                            petsc_vec.data(), &petsc_nullspace);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatNullSpaceCreate");

  // Set null space
  dolfin_assert(_ksp);
  ierr = KSPSetNullSpace(_ksp, petsc_nullspace);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetNullSpace");
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
  return solve(as_type<const PETScBaseMatrix>(A),
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
                 "Non-matching dimensions for linear system (matrix has %d rows and right-hand side vector has %d rows)",
                 _matA->size(0), b.size());
  }

  // Write a message
  const bool report = parameters["report"];
  if (report && dolfin::MPI::rank(PETSC_COMM_WORLD) == 0)
    info("Solving linear system of size %d x %d (PETSc Krylov solver).", M, N);

  // Reinitialize solution vector if necessary
  if (x.empty())
  {
    _matA->init_vector(x, 1);
    x.zero();
  }

  // Set some PETSc-specific options
  set_petsc_ksp_options();

  // Set operators
  set_petsc_operators();

  // Set near null space for preconditioner
  if (_preconditioner)
  {
    dolfin_assert(_matP);
    const MatNullSpace pc_nullspace = _preconditioner->near_nullspace();

    if (pc_nullspace && !preconditioner_set)
    {
      #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 3
      ierr = MatSetNearNullSpace(_matP->mat(), pc_nullspace);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetNearNullSpace");
      #else
      dolfin_error("PETScMatrix.cpp",
                   "set approximate null space for PETSc matrix",
                   "This is supported by PETSc version > 3.2");
      #endif
    }
  }

  // FIXME: Improve check for re-setting preconditoner, e.g. if
  //        parameters change
  // FIXME: Solve using matrix free matrices fails if no user provided
  //        Prec is provided
  // Set preconditioner if necessary
  if (_preconditioner && !preconditioner_set)
  {
    _preconditioner->set(*this);
    preconditioner_set = true;
  }
  // User defined preconditioner
  else if (pc_dolfin && !preconditioner_set)
  {
    PETScUserPreconditioner::setup(_ksp, *pc_dolfin);
    preconditioner_set = true;
  }

  // Check whether we need a work-around for a bug in PETSc-stable.
  // This has been fixed in PETSc-dev, see
  // https://bugs.launchpad.net/dolfin/+bug/988494
  const bool use_petsc_cusp_hack = parameters["use_petsc_cusp_hack"];
  if (use_petsc_cusp_hack)
    info("Using hack to get around PETScCusp bug: ||b|| = %g", b.norm("l2"));

  // Set convergence norm type
  if (parameters["convergence_norm_type"].is_set())
  {
    const std::string convergence_norm_type
      = parameters["convergence_norm_type"];
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

  std::string prefix = std::string(parameters["options_prefix"]);
  if (prefix != "default")
  {
    // Make sure that the prefix has a '_' at the end if the user didn't provide it
    char lastchar = *prefix.rbegin();
    if (lastchar != '_')
      prefix += "_";

    KSPSetOptionsPrefix(_ksp, prefix.c_str());
  }
  KSPSetFromOptions(_ksp);

  // Solve linear system
  if (MPI::rank(PETSC_COMM_WORLD) == 0)
  {
    log(PROGRESS, "PETSc Krylov solver starting to solve %i x %i system.",
        _matA->size(0), _matA->size(1));
  }

  if (parameters["profile"].is_set())
  {
    const bool profile_performance = parameters["profile"];
    if (profile_performance)
    {
      PetscLogBegin();
      ierr = KSPSolve(_ksp, b.vec(), x.vec());
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSolve");
      PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
    }
  }
  else
  {
    ierr =  KSPSolve(_ksp, b.vec(), x.vec());
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSolve");
  }

  // Update ghost values
  x.update_ghost_values();

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = KSPGetIterationNumber(_ksp, &num_iterations);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetIterationNumber");

  // Check if the solution converged and print error/warning if not converged
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
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
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
  if (report && dolfin::MPI::rank(PETSC_COMM_WORLD) == 0)
    write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(const PETScBaseMatrix& A,
                                      PETScVector& x,
                                      const PETScVector& b)
{
  // Set operator
  std::shared_ptr<const PETScBaseMatrix> Atmp(&A, NoDeleter());
  set_operator(Atmp);

  // Call solve
  return solve(x, b);
}
//-----------------------------------------------------------------------------
KSP PETScKrylovSolver::ksp() const
{
  return _ksp;
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
void PETScKrylovSolver::init(const std::string& method)
{
  PetscErrorCode ierr;

  if (_ksp)
    KSPDestroy(&_ksp);

  // Set up solver environment
  ierr = KSPCreate(PETSC_COMM_WORLD, &_ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");

  // Set solver type
  if (method != "default")
  {
    ierr = KSPSetType(_ksp, _methods.find(method)->second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");
  }
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_operators()
{
  dolfin_assert(_matA);
  dolfin_assert(_matP);
  dolfin_assert(_ksp);

  PetscErrorCode ierr;

  // Get parameter
  #if PETSC_VERSION_RELEASE
  const std::string mat_structure = parameters("preconditioner")["structure"];

  // Set operators with appropriate option
  if (mat_structure == "same")
  {
    ierr = KSPSetOperators(_ksp, _matA->mat(), _matP->mat(), SAME_PRECONDITIONER);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
  }
  else if (mat_structure == "same_nonzero_pattern")
  {
    ierr = KSPSetOperators(_ksp, _matA->mat(), _matP->mat(), SAME_NONZERO_PATTERN);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
  }
  else if (mat_structure == "different_nonzero_pattern")
  {
    ierr = KSPSetOperators(_ksp, _matA->mat(), _matP->mat(),
                           DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
  }
  else
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "set PETSc Krylov solver operators",
                 "Preconditioner re-use paramrter \"%s \" is unknown",
                 mat_structure.c_str());
  }
  #else
  ierr = KSPSetOperators(_ksp, _matA->mat(), _matP->mat());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
  #endif
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_ksp_options()
{
  PetscErrorCode ierr;

  // GMRES restart parameter
  const int gmres_restart = parameters("gmres")["restart"];
  ierr = KSPGMRESSetRestart(_ksp, gmres_restart);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGMRESSetRestart");

  // Non-zero initial guess
  const bool nonzero_guess = parameters["nonzero_initial_guess"];
  PetscBool petsc_nonzero_guess = PETSC_FALSE;
  if (nonzero_guess)
    petsc_nonzero_guess = PETSC_TRUE;
  ierr = KSPSetInitialGuessNonzero(_ksp, petsc_nonzero_guess);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetInitialGuessNonzero");

  // Monitor convergence
  const bool monitor_convergence = parameters["monitor_convergence"];
  if (monitor_convergence)
  {
    ierr = KSPMonitorSet(_ksp, KSPMonitorTrueResidualNorm, 0, 0);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPMonitorSet");
  }

  // Set tolerances
  const int max_iterations = parameters["maximum_iterations"];
  ierr = KSPSetTolerances(_ksp,
                          parameters["relative_tolerance"],
                          parameters["absolute_tolerance"],
                          parameters["divergence_limit"],
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
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 3
  const KSPType ksp_type;
  const PCType pc_type;
  #else
  KSPType ksp_type;
  PCType pc_type;
  #endif
  ierr = KSPGetType(_ksp, &ksp_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetType");

  ierr = KSPGetPC(_ksp, &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  ierr = PCGetType(pc, &pc_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCGetType");

  // If using additive Schwarz or block Jacobi, get 'sub' method which is
  // applied to each block
  const std::string pc_type_str = pc_type;
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 3
  const KSPType sub_ksp_type;
  const PCType sub_pc_type;
  #else
  KSPType sub_ksp_type;
  PCType sub_pc_type;
  #endif
  PC sub_pc;
  KSP* sub_ksp = NULL;
  if (pc_type_str == PCASM || pc_type_str == PCBJACOBI)
  {
    if (pc_type_str == PCASM)
    {
      ierr = PCASMGetSubKSP(pc, PETSC_NULL, PETSC_NULL, &sub_ksp);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCASMGetSubKSP");
    }
    else if (pc_type_str == PCBJACOBI)
    {
      ierr = PCBJacobiGetSubKSP(pc, PETSC_NULL, PETSC_NULL, &sub_ksp);
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
                 "Non-matching dimensions for linear system (matrix has %d rows and right-hand side vector has %d rows)",
                 A.size(0), b.size());
  }

  // Check dimensions of A vs x
  if (!x.empty() && x.size() != A.size(1))
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "unable to solve linear system with PETSc Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %d columns and solution vector has %d rows)",
                 A.size(1), x.size());
  }

  // FIXME: We could implement a more thorough check of local/global
  // FIXME: dimensions for distributed matrices and vectors here.
}
//-----------------------------------------------------------------------------

#endif
