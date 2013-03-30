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
// Last changed: 2013-03-18

#ifdef HAS_PETSC

#include <petsclog.h>

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

#include <dolfin/common/timing.h>

using namespace dolfin;

// Utility functions
namespace dolfin
{
  class PETScKSPDeleter
  {
  public:
    void operator() (KSP* _ksp)
    {
      if (_ksp)
        KSPDestroy(_ksp);
      delete _ksp;
    }
  };
}

namespace dolfin
{
  class PETScMatNullSpaceDeleter
  {
  public:
    void operator() (MatNullSpace* ns)
    {
      if (*ns)
        MatNullSpaceDestroy(ns);
      delete ns;
    }
  };
}

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
  KSPSetOptionsPrefix(*_ksp, prefix.c_str());
}
//-----------------------------------------------------------------------------
Parameters PETScKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("petsc_krylov_solver");

  // Preconditioing side
  p.add("preconditioner_side", "left");

  // Norm type used in covergence test
  std::set<std::string> allowed_norm_types;
  allowed_norm_types.insert("preconditioned");
  allowed_norm_types.insert("true");
  allowed_norm_types.insert("none");
  p.add("convergence_norm_type", allowed_norm_types);

  // Control PETSc performance profiling
  p.add("profile", false);

  return p;
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     std::string preconditioner)
  : pc_dolfin(0), _preconditioner(new PETScPreconditioner(preconditioner)),
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

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
                                     PETScPreconditioner& preconditioner)
 : _preconditioner(reference_to_no_delete_pointer(preconditioner)),
   preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(std::string method,
  boost::shared_ptr<PETScPreconditioner> preconditioner)
  : _preconditioner(preconditioner), preconditioner_set(false)

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
PETScKrylovSolver::PETScKrylovSolver(std::string method,
  boost::shared_ptr<PETScUserPreconditioner> preconditioner)
  : pc_dolfin(preconditioner.get()), preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(boost::shared_ptr<KSP> ksp)
  : pc_dolfin(0), _ksp(ksp), preconditioner_set(true)
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
void PETScKrylovSolver::set_operator(const boost::shared_ptr<const GenericLinearOperator> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operator(const boost::shared_ptr<const PETScBaseMatrix> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(const boost::shared_ptr<const GenericLinearOperator> A,
                                      const boost::shared_ptr<const GenericLinearOperator> P)
{
  set_operators(as_type<const PETScBaseMatrix>(A),
                as_type<const PETScBaseMatrix>(P));
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(const boost::shared_ptr<const PETScBaseMatrix> A,
                                      const boost::shared_ptr<const PETScBaseMatrix> P)
{
  _A = A;
  _P = P;
  dolfin_assert(_A);
  dolfin_assert(_P);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_nullspace(const std::vector<const GenericVector*> nullspace)
{
  // Copy vectors
  for (std::size_t i = 0; i < nullspace.size(); ++i)
  {
    dolfin_assert(nullspace[i]);
    const PETScVector& x = nullspace[i]->down_cast<PETScVector>();

    // Copy vector
    _nullspace.push_back(x);
  }

  // Get pointers to underlying PETSc objects and normalize vectors
  std::vector<Vec> petsc_vec(nullspace.size());
  for (std::size_t i = 0; i < nullspace.size(); ++i)
  {
    petsc_vec[i] = *(_nullspace[i].vec().get());
    PetscReal val = 0.0;
    VecNormalize(petsc_vec[i], &val);
  }

  // Create null space
  petsc_nullspace.reset(new MatNullSpace, PETScMatNullSpaceDeleter());
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, nullspace.size(),
                     petsc_vec.data(), petsc_nullspace.get());

  // Set null space
  dolfin_assert(_ksp);
  KSPSetNullSpace(*_ksp, *petsc_nullspace);
}
//-----------------------------------------------------------------------------
const PETScBaseMatrix& PETScKrylovSolver::get_operator() const
{
  if (!_A)
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "access operator for PETSc Krylov solver",
                 "Operator has not been set");
  }
  return *_A;
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  //check_dimensions(*A, x, b);
  return solve(as_type<PETScVector>(x), as_type<const PETScVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(const GenericLinearOperator& A,
                                      GenericVector& x,
                                      const GenericVector& b)
{
  //check_dimensions(A, x, b);
  return solve(as_type<const PETScBaseMatrix>(A),
               as_type<PETScVector>(x),
               as_type<const PETScVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(PETScVector& x, const PETScVector& b)
{
  dolfin_assert(_A);
  dolfin_assert(_ksp);

  // Check dimensions
  const std::size_t M = _A->size(0);
  const std::size_t N = _A->size(1);
  if (_A->size(0) != b.size())
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "unable to solve linear system with PETSc Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %d rows and right-hand side vector has %d rows)",
                 _A->size(0), b.size());
  }

  // Write a message
  const bool report = parameters["report"];
  if (report && dolfin::MPI::process_number() == 0)
    info("Solving linear system of size %d x %d (PETSc Krylov solver).", M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    _A->resize(x, 1);
    x.zero();
  }

  // Set some PETSc-specific options
  set_petsc_options();

  // Set operators
  set_petsc_operators();

  // Set (approxinate) null space for preconditioner
  if (_preconditioner)
  {
    dolfin_assert(_P);
    boost::shared_ptr<const MatNullSpace> pc_nullspace = _preconditioner->nullspace();
    if (pc_nullspace)
    {
      #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 3
      MatSetNearNullSpace(*(_P->mat()), *pc_nullspace);
      #else
      dolfin_error("PETScMatrix.cpp",
                   "set approximate null space for PETSc matrix",
                   "This is supported by PETSc version > 3.2");
      #endif
    }
  }

  // FIXME: Improve check for re-setting preconditoner, e.g. if parameters change
  // FIXME: Solve using matrix free matrices fails if no user provided Prec is provided
  // Set preconditioner if necessary
  if (_preconditioner && !preconditioner_set)
  {
    _preconditioner->set(*this);
    preconditioner_set = true;
  }

  // User defined preconditioner
  else if (pc_dolfin && !preconditioner_set)
  {
    PETScUserPreconditioner::setup(*_ksp, *pc_dolfin);
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
    const std::string convergence_norm_type = parameters["convergence_norm_type"];
    if (convergence_norm_type == "true")
      KSPSetNormType(*_ksp, KSP_NORM_UNPRECONDITIONED);
    else if (convergence_norm_type == "preconditioned")
      KSPSetNormType(*_ksp, KSP_NORM_PRECONDITIONED);
    else if (convergence_norm_type == "none")
      KSPSetNormType(*_ksp, KSP_NORM_NONE);
  }

  // Solve linear system
  if (MPI::process_number() == 0)
  {
    log(PROGRESS, "PETSc Krylov solver starting to solve %i x %i system.",
        _A->size(0), _A->size(1));
  }

  const bool profile_performance = parameters["profile"];
  if (profile_performance)
  {
    PetscLogBegin();
    KSPSolve(*_ksp, *b.vec(), *x.vec());
    PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  }
  else
    KSPSolve(*_ksp, *b.vec(), *x.vec());

  // Get the number of iterations
  PetscInt num_iterations = 0;
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
  if (report && dolfin::MPI::process_number() == 0)
    write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(const PETScBaseMatrix& A,
                                      PETScVector& x,
                                      const PETScVector& b)
{
  // Set operator
  boost::shared_ptr<const PETScBaseMatrix> Atmp(&A, NoDeleter());
  set_operator(Atmp);

  // Call solve
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
  {
    dolfin_error("PETScKrylovSolver.cpp",
                 "initialize PETSc Krylov solver",
                 "More than one object points to the underlying PETSc object");
  }

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
    KSPSetType(*_ksp, _methods.find(method)->second);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_operators()
{
  dolfin_assert(_A);
  dolfin_assert(_P);

  // Get some parameters
  const bool reuse_precon = parameters("preconditioner")["reuse"];
  const bool same_pattern = parameters("preconditioner")["same_nonzero_pattern"];

  // Set operators with appropriate option
  if (reuse_precon)
    KSPSetOperators(*_ksp, *_A->mat(), *_P->mat(), SAME_PRECONDITIONER);
  else if (same_pattern)
    KSPSetOperators(*_ksp, *_A->mat(), *_P->mat(), SAME_NONZERO_PATTERN);
  else
    KSPSetOperators(*_ksp, *_A->mat(), *_P->mat(), DIFFERENT_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_petsc_options()
{
  // GMRES restart parameter
  const int gmres_restart = parameters("gmres")["restart"];
  KSPGMRESSetRestart(*_ksp, gmres_restart);

  // Non-zero initial guess
  const bool nonzero_guess = parameters["nonzero_initial_guess"];
  if (nonzero_guess)
    KSPSetInitialGuessNonzero(*_ksp, PETSC_TRUE);
  else
    KSPSetInitialGuessNonzero(*_ksp, PETSC_FALSE);

  // Monitor convergence
  const bool monitor_convergence = parameters["monitor_convergence"];
  if (monitor_convergence)
    KSPMonitorSet(*_ksp, KSPMonitorTrueResidualNorm, 0, 0);

  // Set tolerances
  const int max_iterations = parameters["maximum_iterations"];
  KSPSetTolerances(*_ksp,
                   parameters["relative_tolerance"],
                   parameters["absolute_tolerance"],
                   parameters["divergence_limit"],
                   max_iterations);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::write_report(int num_iterations,
                                     KSPConvergedReason reason)
{
  // Get name of solver and preconditioner
  PC pc;
  #if PETSC_VERSION_RELEASE
  const KSPType ksp_type;
  const PCType pc_type;
  #else
  KSPType ksp_type;
  PCType pc_type;
  #endif
  KSPGetType(*_ksp, &ksp_type);
  KSPGetPC(*_ksp, &pc);
  PCGetType(pc, &pc_type);

  // If using additive Schwarz or block Jacobi, get 'sub' method which is
  // applied to each block
  const std::string pc_type_str = pc_type;
  #if PETSC_VERSION_RELEASE
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
      PCASMGetSubKSP(pc, PETSC_NULL, PETSC_NULL, &sub_ksp);
    else if (pc_type_str == PCBJACOBI)
      PCBJacobiGetSubKSP(pc, PETSC_NULL, PETSC_NULL, &sub_ksp);
    KSPGetType(*sub_ksp, &sub_ksp_type);
    KSPGetPC(*sub_ksp, &sub_pc);
    PCGetType(sub_pc, &sub_pc_type);
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
    PCHYPREGetType(pc, &hypre_sub_type);

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
