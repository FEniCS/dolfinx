// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
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

#include <petscksp.h>
#include <petscpc.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "PETScMatrix.h"
#include "PETScObject.h"
#include "PETScVector.h"
#include "PETScLUSolver.h"

using namespace dolfin;

// Functions in anonymous namespace (local scope)
namespace
{
  const MatSolverPackage get_solver_package_type(KSP ksp)
  {
    PetscErrorCode ierr;
    const MatSolverPackage solver_type;
    PC pc;
    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) dolfin::PETScObject::petsc_error(ierr, __FILE__, "KSPGetPC");

    ierr = PCFactorGetMatSolverPackage(pc, &solver_type);
    if (ierr != 0) dolfin::PETScObject::petsc_error(ierr, __FILE__, "PCFactorGetMatSolverPackage");

    return solver_type;
  }
  //---------------------------------------------------------------------------
  std::map<const MatSolverPackage, bool> methods_cholesky
  = { {MATSOLVERUMFPACK,      false},
      {MATSOLVERMUMPS,        true},
      {MATSOLVERPASTIX,       true},
      {MATSOLVERSUPERLU,      false},
      {MATSOLVERSUPERLU_DIST, false},
      {MATSOLVERPETSC,        true} };

  //---------------------------------------------------------------------------
  bool solver_has_cholesky(const MatSolverPackage package)
  {
    auto it = methods_cholesky.find(package);
    dolfin_assert(it != methods_cholesky.end());
    return it->second;
  }
  //---------------------------------------------------------------------------
  const std::map<std::string, std::string> methods_descr
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

}
//-----------------------------------------------------------------------------

// List of available LU solvers
std::map<std::string, const MatSolverPackage> PETScLUSolver::lumethods
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
std::map<std::string, std::string> PETScLUSolver::methods()
{
  return methods_descr;
}
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
                             std::string method) : _solver(comm)
{
  PetscErrorCode ierr;

  // Check dimensions, and check for symmetry
  PetscBool is_symmetric = PETSC_FALSE;
  if (A)
  {
    if (A->size(0) != A->size(1))
    {
      dolfin_error("PETScLUSolver.cpp",
                   "create PETSc LU solver",
                   "Cannot LU factorize non-square PETSc matrix");
    }

    dolfin_assert(A->mat());
    PetscBool symm_is_set = PETSC_FALSE;
    ierr = MatIsSymmetricKnown(A->mat(), &symm_is_set, &is_symmetric);
    if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "MatIsSymmetricKnown");
  }

  // Select solver package
  const MatSolverPackage solver_package = select_solver(comm, method);

  // Get KSP pointer
  KSP ksp = _solver.ksp();

  // Make solver preconditioner only
  ierr = KSPSetType(ksp, KSPPREONLY);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "KSPSetType");

  // Get PC
  PC pc;
  ierr = KSPGetPC(ksp, &pc);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "KSPGetPC");

  // Set PC type to LU or PCCHOLESKY (depending on matrix symmetry)
  if (is_symmetric == PETSC_TRUE and solver_has_cholesky(solver_package))
  {
    ierr = PCSetType(pc, PCCHOLESKY);
    if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PCSetType");
  }
  else
  {
    ierr = PCSetType(pc, PCLU);
    if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PCSetType");
  }

  // Set LU solver package
  ierr = PCFactorSetMatSolverPackage(pc, solver_package);
  if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "PCFactorSetMatSolverPackage");

  // Set operator
  if (A)
  {
    ierr = KSPSetOperators(ksp, A->mat(), A->mat());
    if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "KSPSetOperators");
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
void PETScLUSolver::set_operator(const PETScMatrix& A)
{
  _solver.set_operator(A);
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
  // FIXME: This should really go in PETScKrylovSolver
  /*
  const bool report = parameters["report"].is_set() ? parameters["report"] : false;
  if (report && dolfin::MPI::rank(mpi_comm()) == 0)
  {
    // Get PETSc operators
    Mat _A, _P;
    KSPGetOperators(_solver.ksp(), &_A, &_P);
    dolfin_assert(_A);
    PETScBaseMatrix A(_A);

    const MatSolverPackage solver_type = get_solver_package_type(_solver.ksp());
    log(PROGRESS,"Solving linear system of size %ld x %ld (PETSc LU solver, %s).",
        A.size(0), A.size(1), solver_type);
  }
  */
  return _solver.solve(x, b);
}
//-----------------------------------------------------------------------------
std::size_t PETScLUSolver::solve(const PETScMatrix& A, PETScVector& x,
                                 const PETScVector& b)
{
  _solver.set_operators(A, A);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_options_prefix(std::string options_prefix)
{
  _solver.set_options_prefix(options_prefix);
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::get_options_prefix() const
{
  return _solver.get_options_prefix();
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_from_options() const
{
  _solver.set_from_options();
}
//-----------------------------------------------------------------------------
MPI_Comm PETScLUSolver::mpi_comm() const
{
  return _solver.mpi_comm();
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScLUSolver not implemented, calling PETSc KSPView directly.");
    PetscErrorCode ierr = KSPView(_solver.ksp(), PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) PETScObject::petsc_error(ierr, __FILE__, "KSPView");
  }
  else
    s << "<PETScLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
KSP PETScLUSolver::ksp() const
{
  return _solver.ksp();
}
//-----------------------------------------------------------------------------
const MatSolverPackage PETScLUSolver::select_solver(MPI_Comm comm,
                                                    std::string method)
{
  // Check package string
  if (lumethods.count(method) == 0)
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

  auto it = lumethods.find(method);
  dolfin_assert(it !=  lumethods.end());
  return it->second;
}
//-----------------------------------------------------------------------------

#endif
