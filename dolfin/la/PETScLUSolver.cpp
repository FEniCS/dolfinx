// Copyright (C) 2005-2011 Anders Logg
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
// Modified by Garth N. Wells 2009-2010
// Modified by Niclas Jansson 2009
// Modified by Fredrik Valdmanis 2011
//
// First added:  2005
// Last changed: 2011-11-11

#ifdef HAS_PETSC

#include <dolfin/common/constants.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "LUSolver.h"
#include "PETScMatrix.h"
#include "PETScOptions.h"
#include "PETScVector.h"
#include "PETScLUSolver.h"

using namespace dolfin;

// List of available LU solvers
const std::map<std::string, const MatSolverPackage> PETScLUSolver::_methods
= { {"default", ""},
#if PETSC_HAVE_UMFPACK || PETSC_HAVE_SUITESPARSE
    {"umfpack",      MATSOLVERUMFPACK},
#endif
#if PETSC_HAVE_MUMPS
    {"mumps",        MATSOLVERMUMPS},
#endif
#if PETSC_HAVE_PASTIX
    {"pastix",       MATSOLVERPASTIX},
#endif
#if PETSC_HAVE_SUPERLU
    {"superlu",      MATSOLVERSUPERLU},
#endif
#if PETSC_HAVE_SUPERLU_DIST
    {"superlu_dist", MATSOLVERSUPERLU_DIST},
#endif
    {"petsc",        MATSOLVERPETSC}};
//-----------------------------------------------------------------------------
const std::map<const MatSolverPackage, const bool>
PETScLUSolver::_methods_cholesky
= { {MATSOLVERUMFPACK,      false},
    {MATSOLVERMUMPS,        true},
    {MATSOLVERPASTIX,       true},
    {MATSOLVERSUPERLU,      false},
    {MATSOLVERSUPERLU_DIST, false},
    {MATSOLVERPETSC,        true} };
//-----------------------------------------------------------------------------
const std::map<std::string, std::string>
PETScLUSolver::_methods_descr
= { {"default", "default LU solver"},
#if PETSC_HAVE_UMFPACK || PETSC_HAVE_SUITESPARSE
    {"umfpack", "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)"},
#endif
#if PETSC_HAVE_MUMPS
    {"mumps", "MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)"},
#endif
#if PETSC_HAVE_PASTIX
    {"pastix", "PaStiX (Parallel Sparse matriX package)"},
#endif
#if PETSC_HAVE_SUPERLU
    {"superlu", "SuperLU"},
#endif
#if PETSC_HAVE_SUPERLU_DIST
    {"superlu_dist", "Parallel SuperLU"},
#endif
    {"petsc", "PETSc built in LU solver"} };

//-----------------------------------------------------------------------------
std::map<std::string, std::string>
PETScLUSolver::methods()
{
  return PETScLUSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
Parameters PETScLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("petsc_lu_solver");

  // Number of threads per process for multi-threaded solvers
  p.add<std::size_t>("num_threads");

  return p;
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(MPI_Comm comm, std::string method) : _ksp(NULL)
{
  // Set parameter values
  parameters = default_parameters();

  // Initialize PETSc LU solver
  init_solver(comm, method);
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(std::string method)
  : PETScLUSolver(MPI_COMM_WORLD, method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(MPI_Comm comm,
                             std::shared_ptr<const PETScMatrix> A,
                             std::string method) : _ksp(NULL), _matA(A)
{
  // Check dimensions
  if (A->size(0) != A->size(1))
  {
    dolfin_error("PETScLUSolver.cpp",
                 "create PETSc LU solver",
                 "Cannot LU factorize non-square PETSc matrix");
  }

  // Set parameter values
  parameters = default_parameters();

  // Initialize PETSc LU solver
  init_solver(comm, method);
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(std::shared_ptr<const PETScMatrix> A,
                             std::string method)
  : PETScLUSolver(MPI_COMM_WORLD, A, method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScLUSolver::~PETScLUSolver()
{
  if (_ksp)
    KSPDestroy(&_ksp);
}
//-----------------------------------------------------------------------------
void
PETScLUSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  // Attempt to cast as PETScMatrix
  std::shared_ptr<const PETScMatrix> mat
    = as_type<const PETScMatrix>(require_matrix(A));
  dolfin_assert(mat);

  // Set operator
  set_operator(mat);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_operator(std::shared_ptr<const PETScMatrix> A)
{
  _matA = A;
  dolfin_assert(_matA);
  dolfin_assert(_ksp);

  if (!_matA->mat())
  {
    dolfin_error("PETScLUSolver.cpp",
                 "set operator (PETScLUSolver::set_operator)",
                 "cannot set operator if matrix has not been initialized");
  }

  PetscErrorCode ierr;
  ierr = KSPSetOperators(_ksp, _matA->mat(), _matA->mat());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
const GenericLinearOperator& PETScLUSolver::get_operator() const
{
  if (!_matA)
  {
    dolfin_error("PETScLUSolver.cpp",
                 "access operator of PETSc LU solver",
                 "Operator has not been set");
  }
  return *_matA;
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  return solve(x, b, false);
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve(GenericVector& x, const GenericVector& b,
                                 bool transpose)
{
  Timer timer("PETSc LU solver");

  dolfin_assert(_ksp);
  dolfin_assert(_matA);
  PetscErrorCode ierr;

  // Downcast matrix and vectors
  const PETScVector& _b = as_type<const PETScVector>(b);
  PETScVector& _x = as_type<PETScVector>(x);

  // Check dimensions
  if (_matA->size(0) != b.size())
  {
    dolfin_error("PETScLUSolver.cpp",
                 "solve linear system using PETSc LU solver",
                 "Cannot factorize non-square PETSc matrix");
  }

  // Initialize solution vector if required (make compatible with A in
  // parallel)
  if (x.empty())
    _matA->init_vector(x, 1);

  // Set PETSc operators (depends on factorization re-use options);
  //set_petsc_operators();

  // Write a pre-solve message
  pre_report(*_matA);

  // Get package used to solve system
  PC pc;
  ierr = KSPGetPC(_ksp, &pc);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPGetPC");

  configure_ksp(_solver_package);

  // Set number of threads if using PaStiX
  if (strcmp(_solver_package, MATSOLVERPASTIX) == 0)
  {
    const std::size_t num_threads = parameters["num_threads"].is_set() ?
      parameters["num_threads"] : dolfin::parameters["num_threads"];
    PETScOptions::set("-mat_pastix_threadnbr", num_threads);
  }

  // Solve linear system
  const Vec b_petsc = _b.vec();
  Vec x_petsc = _x.vec();
  if (!transpose)
  {
    ierr = KSPSolve(_ksp, b_petsc, x_petsc);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSolve");
  }
  else
  {
    ierr = KSPSolveTranspose(_ksp, b_petsc, x_petsc);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSolveTranspose");
  }

  // Update ghost values following solve
  _x.update_ghost_values();

  return 1;
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve(const GenericLinearOperator& A,
                                 GenericVector& x,
                                 const GenericVector& b)
{
  return solve(as_type<const PETScMatrix>(require_matrix(A)),
               as_type<PETScVector>(x),
               as_type<const PETScVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve(const PETScMatrix& A, PETScVector& x,
                                 const PETScVector& b)
{
  std::shared_ptr<const PETScMatrix> Atmp(&A, NoDeleter());
  set_operator(Atmp);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve_transpose(GenericVector& x,
                                           const GenericVector& b)
{
return solve(x, b, true);
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve_transpose(const GenericLinearOperator& A,
                                           GenericVector& x,
                                           const GenericVector& b)
{
  return solve_transpose(as_type<const PETScMatrix>(require_matrix(A)),
                         as_type<PETScVector>(x),
                         as_type<const PETScVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve_transpose(const PETScMatrix& A,
                                           PETScVector& x,
                                           const PETScVector& b)
{
  std::shared_ptr<const PETScMatrix> _matA(&A, NoDeleter());
  set_operator(_matA);
  return solve_transpose(x, b);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_options_prefix(std::string options_prefix)
{
  if (_ksp)
  {
    dolfin_error("PETScLUSolver.cpp",
                 "setting PETSc options prefix",
                 "Cannot set options prefix since PETSc KSP has already been initialized");
  }
  else
    _petsc_options_prefix = options_prefix;
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::get_options_prefix() const
{
  if (_ksp)
  {
    const char* prefix = NULL;
    KSPGetOptionsPrefix(_ksp, &prefix);
    return std::string(prefix);
  }
  else
  {
    warning("PETSc KSP object has not been initialised, therefore prefix has not been set");
    return std::string();
  }
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_from_options() const
{
  dolfin_assert(_ksp);
  PetscErrorCode ierr = KSPSetFromOptions(_ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetFromOptions");
}
//-----------------------------------------------------------------------------
MPI_Comm PETScLUSolver::mpi_comm() const
{
  dolfin_assert(_ksp);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_ksp, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScLUSolver not implemented, calling PETSc KSPView directly.");
    PetscErrorCode ierr = KSPView(_ksp, PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPView");
  }
  else
    s << "<PETScLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
KSP PETScLUSolver::ksp() const
{
  return _ksp;
}
//-----------------------------------------------------------------------------
const MatSolverPackage PETScLUSolver::select_solver(std::string& method) const
{
  // Check package string
  if (_methods.count(method) == 0)
  {
    dolfin_error("PETScLUSolver.cpp",
                 "solve linear system using PETSc LU solver",
                 "Unknown LU method \"%s\"", method.c_str());
  }

  // Choose appropriate 'default' solver
  if (method == "default")
  {
    #if defined(PETSC_USE_64BIT_INDICES)
    if (MPI::size(mpi_comm()) == 1)
    {
      #if PETSC_HAVE_UMFPACK || PETSC_HAVE_SUITESPARSE
      method = "umfpack";
      #elif PETSC_HAVE_SUPERLU_DIST
      method = "superlu_dist";
      #else
      method = "petsc";
      warning("Using PETSc native LU solver. Consider configuring PETSc with an efficient LU solver (e.g. Umfpack, SuperLU_dist).");
      #endif
    }
    else
    {
      #if PETSC_HAVE_SUPERLU_DIST
      method = "superlu_dist";
      #else
      method = "petsc";
      warning("Using PETSc native LU solver. Consider configuring PETSc with an efficient LU solver (e.g. SuperLU_dist).");
      #endif

    }
    #else
    if (MPI::size(mpi_comm()) == 1)
    {
      #if PETSC_HAVE_UMFPACK || PETSC_HAVE_SUITESPARSE
      method = "umfpack";
      #elif PETSC_HAVE_MUMPS
      method = "mumps";
      #elif PETSC_HAVE_PASTIX
      method = "pastix";
      #elif PETSC_HAVE_SUPERLU
      method = "superlu";
      #elif PETSC_HAVE_SUPERLU_DIST
      method = "superlu_dist";
      #else
      method = "petsc";
      warning("Using PETSc native LU solver. Consider configuring PETSc with an efficient LU solver (e.g. UMFPACK, MUMPS).");
      #endif
    }
    else
    {
      #if PETSC_HAVE_MUMPS
      method = "mumps";
      #elif PETSC_HAVE_SUPERLU_DIST
      method = "superlu_dist";
      #elif PETSC_HAVE_PASTIX
      method = "pastix";
      #else
      dolfin_error("PETScLUSolver.cpp",
                   "solve linear system using PETSc LU solver",
                   "No suitable solver for parallel LU found. Consider configuring PETSc with MUMPS or SuperLU_dist");
      #endif
    }
    #endif
  }

  return _methods.find(method)->second;
}
//-----------------------------------------------------------------------------
bool PETScLUSolver::solver_has_cholesky(const MatSolverPackage package) const
{
  return _methods_cholesky.find(package)->second;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::init_solver(MPI_Comm comm, std::string& method)
{
  // Destroy old solver environment if necessary
  if (_ksp)
    KSPDestroy(&_ksp);

  PetscErrorCode ierr;

  // Create solver
  ierr = KSPCreate(comm, &_ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");

  // Select solver (must come after KSPCreate, becuase we get the MPI
  // communicator from the KSO object)
  _solver_package = select_solver(method);

  // Set options prefix (if any)
  ierr = KSPSetOptionsPrefix(_ksp, _petsc_options_prefix.c_str());
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOptionsPrefix");

  // Make solver preconditioner only
  ierr = KSPSetType(_ksp, KSPPREONLY);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");

  // Set from PETSc options
  KSPSetFromOptions(_ksp);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::configure_ksp(const MatSolverPackage solver_package)
{
  PetscErrorCode ierr;

  PC pc;
  ierr = KSPGetPC(_ksp, &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  // Set preconditioner to LU factorization/Cholesky as appropriate

  const bool symmetric = parameters["symmetric"].is_set() ? parameters["symmetric"] : false;
  if (symmetric and solver_has_cholesky(solver_package))
  {
    ierr = PCSetType(pc, PCCHOLESKY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
  }
  else
  {
    ierr = PCSetType(pc, PCLU);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
  }

  // Set solver package
  ierr = PCFactorSetMatSolverPackage(pc, solver_package);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetMatSolverPackage");

  // Allow matrices with zero diagonals to be solved
  ierr = PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetShiftType");
  ierr = PCFactorSetShiftAmount(pc, PETSC_DECIDE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetShiftAmount");
}
//-----------------------------------------------------------------------------
void PETScLUSolver::pre_report(const PETScMatrix& A) const
{
  PetscErrorCode ierr;

  const MatSolverPackage solver_type;
  PC pc;
  ierr = KSPGetPC(_ksp, &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  ierr = PCFactorGetMatSolverPackage(pc, &solver_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorGetMatSolverPackage");

  // Get parameter
  const bool report = parameters["report"].is_set() ? parameters["report"] : false;
  if (report && dolfin::MPI::rank(mpi_comm()) == 0)
  {
    log(PROGRESS,"Solving linear system of size %ld x %ld (PETSc LU solver, %s).",
        A.size(0), A.size(1), solver_type);
  }
}
//-----------------------------------------------------------------------------
#endif
