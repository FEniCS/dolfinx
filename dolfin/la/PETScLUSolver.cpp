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

#ifdef HAS_PETSC

#include <petscpc.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "LUSolver.h"
#include "PETScMatrix.h"
#include "PETScOptions.h"
#include "PETScVector.h"
#include "PETScLUSolver.h"

using namespace dolfin;

std::map<std::string, const MatSolverPackage> PETScLUSolver::_lumethods
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


// List of available LU solvers
namespace
{
std::map<std::string, const MatSolverPackage> lumethods
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

  //---------------------------------------------------------------------------
  std::map<const MatSolverPackage, bool> methods_cholesky
  = { {MATSOLVERUMFPACK,      false},
      {MATSOLVERMUMPS,        true},
      {MATSOLVERPASTIX,       true},
      {MATSOLVERSUPERLU,      false},
      {MATSOLVERSUPERLU_DIST, false},
      {MATSOLVERPETSC,        true} };
}
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
std::map<std::string, std::string> PETScLUSolver::methods()
{
  return PETScLUSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
//std::map<std::string, const MatSolverPackage> PETScLUSolver::petsc_methods()
//{
//  return _lumethods;
//}
//-----------------------------------------------------------------------------
/*
Parameters PETScLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("petsc_lu_solver");

  // Number of threads per process for multi-threaded solvers
  p.add<std::size_t>("num_threads");

  return p;
}
*/
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(MPI_Comm comm, std::string method)
  :  PETScLUSolver(comm, nullptr, method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(std::string method)
  : PETScLUSolver(MPI_COMM_WORLD, nullptr, method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(MPI_Comm comm,
                             std::shared_ptr<const PETScMatrix> A,
                             std::string method) : PETScKrylovSolver(comm)
{
  // Check dimensions
  if (A)
  {
    if (A->size(0) != A->size(1))
    {
      dolfin_error("PETScLUSolver.cpp",
                   "create PETSc LU solver",
                   "Cannot LU factorize non-square PETSc matrix");
    }
  }

  // Set parameter values
  //parameters = default_parameters();

  // Get KSP pointer
  KSP ksp = PETScKrylovSolver::ksp();

  PetscErrorCode ierr;

  // Get PC
  PC pc;
  ierr = KSPGetPC(ksp, &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  // Create solver
  //ierr = KSPCreate(comm, &ksp);
  //if (ierr != 0) petsc_error(ierr, __FILE__, "KSPCreate");

  // Set preconditioner to LU factorization/Cholesky as appropriate
  //const bool symmetric = parameters["symmetric"].is_set() ? parameters["symmetric"] : false;
  //if (symmetric and solver_has_cholesky(solver_package))
  //
  //  ierr = PCSetType(pc, PCCHOLESKY);
  //  if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
  //}
  //else
  //{
  ierr = PCSetType(pc, PCLU);
  //  if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
  //}

  // Select solver
  const MatSolverPackage solver_package = select_solver(comm, method);

  // Set solver package
  ierr = PCFactorSetMatSolverPackage(pc, solver_package);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetMatSolverPackage");

  // Make solver preconditioner only
  ierr = KSPSetType(_ksp, KSPPREONLY);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");

  // Set operator
  if (A)
  {
    ierr = KSPSetOperators(ksp, A->mat(), A->mat());
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetOperators");
  }
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
  // Do nothing
}
//-----------------------------------------------------------------------------
void
PETScLUSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  PETScKrylovSolver::set_operator(A);
}
//-----------------------------------------------------------------------------
void
PETScLUSolver::set_operator(std::shared_ptr<const PETScMatrix> A)
{
  PETScKrylovSolver::set_operator(A);
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
  return PETScKrylovSolver::solve(x, b);
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
  return PETScKrylovSolver::solve(A, x, b);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_options_prefix(std::string options_prefix)
{
  PETScKrylovSolver::set_options_prefix(options_prefix);
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::get_options_prefix() const
{
  return PETScKrylovSolver::get_options_prefix();
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_from_options() const
{
  PETScKrylovSolver::set_from_options();
}
//-----------------------------------------------------------------------------
MPI_Comm PETScLUSolver::mpi_comm() const
{
  return PETScKrylovSolver::mpi_comm();
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScLUSolver not implemented, calling PETSc KSPView directly.");
    PetscErrorCode ierr = KSPView(ksp(), PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPView");
  }
  else
    s << "<PETScLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
KSP PETScLUSolver::ksp() const
{
  return PETScKrylovSolver::ksp();
}
//-----------------------------------------------------------------------------
const MatSolverPackage PETScLUSolver::select_solver(MPI_Comm comm,
                                                    std::string method)
{
  // Check package string
  if (_lumethods.count(method) == 0)
  {
    dolfin_error("PETScLUSolver.cpp",
                 "solve linear system using PETSc LU solver",
                 "Unknown LU method \"%s\"", method.c_str());
  }

  // Choose appropriate 'default' solver
  if (method == "default")
  {
    #if defined(PETSC_USE_64BIT_INDICES)
    if (dolfin::MPI::size(comm) == 1)
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
    if (dolfin::MPI::size(comm) == 1)
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

  auto it = _lumethods.find(method);
  dolfin_assert(it !=  _lumethods.end());
  return it->second;
}
//-----------------------------------------------------------------------------
/*
bool PETScLUSolver::solver_has_cholesky(const MatSolverPackage package)
{
  auto it = _methods_cholesky.find(package);
  dolfin_assert(it != _methods_cholesky.end());
  return it->second;
}
*/
//-----------------------------------------------------------------------------
/*
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
*/
//-----------------------------------------------------------------------------
#endif
