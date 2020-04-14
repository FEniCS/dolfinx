// Copyright (C) 2014 Johan Jansson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScKrylovSolver.h"
#include "PETScOperator.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <petsclog.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(MPI_Comm comm) : _ksp(nullptr)
{
  PetscErrorCode ierr;

  // Create PETSc KSP object
  ierr = KSPCreate(comm, &_ksp);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPCreate");
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(KSP ksp, bool inc_ref_count) : _ksp(ksp)
{
  assert(_ksp);
  if (inc_ref_count)
  {
    PetscErrorCode ierr = PetscObjectReference((PetscObject)_ksp);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::~PETScKrylovSolver()
{
  if (_ksp)
    KSPDestroy(&_ksp);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operator(const Mat A) { set_operators(A, A); }
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(const Mat A, const Mat P)
{
  assert(A);
  assert(_ksp);
  PetscErrorCode ierr;
  ierr = KSPSetOperators(_ksp, A, P);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
int PETScKrylovSolver::solve(Vec x, const Vec b, bool transpose)
{
  common::Timer timer("PETSc Krylov solver");
  assert(x);
  assert(b);

  // Get PETSc operators
  Mat _A, _P;
  KSPGetOperators(_ksp, &_A, &_P);
  assert(_A);

  // Create wrapper around PETSc Mat object
  // la::PETScOperator A(_A);

  PetscErrorCode ierr;

  // // Check dimensions
  // const std::array<std::int64_t, 2> size = A.size();
  // if (size[0] != b.size())
  // {
  //   log::dolfin_error(
  //       "PETScKrylovSolver.cpp",
  //       "unable to solve linear system with PETSc Krylov solver",
  //       "Non-matching dimensions for linear system (matrix has %ld "
  //       "rows and right-hand side vector has %ld rows)",
  //       size[0], b.size());
  // }

  // Solve linear system
  LOG(INFO) << "PETSc Krylov solver starting to solve system.";

  // Solve system
  if (!transpose)
  {
    ierr = KSPSolve(_ksp, b, x);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPSolve");
  }
  else
  {
    ierr = KSPSolveTranspose(_ksp, b, x);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPSolve");
  }

  // FIXME: Remove ghost updating?
  // Update ghost values in solution vector
  Vec xg;
  VecGhostGetLocalForm(x, &xg);
  const bool is_ghosted = xg ? true : false;
  VecGhostRestoreLocalForm(x, &xg);
  if (is_ghosted)
  {
    VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
  }

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = KSPGetIterationNumber(_ksp, &num_iterations);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPGetIterationNumber");

  // Check if the solution converged and print error/warning if not
  // converged
  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(_ksp, &reason);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPGetConvergedReason");
  if (reason < 0)
  {
    /*
    // Get solver residual norm
    double rnorm = 0.0;
    ierr = KSPGetResidualNorm(_ksp, &rnorm);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetResidualNorm");
    const char *reason_str = KSPConvergedReasons[reason];
    bool error_on_nonconvergence =
    this->parameters["error_on_nonconvergence"].is_set() ?
    this->parameters["error_on_nonconvergence"] : true;
    if (error_on_nonconvergence)
    {
      log::dolfin_error("PETScKrylovSolver.cpp",
                   "solve linear system using PETSc Krylov solver",
                   "Solution failed to converge in %i iterations (PETSc reason
    %s, residual norm ||r|| = %e)",
                   static_cast<int>(num_iterations), reason_str, rnorm);
    }
    else
    {
      log::warning("Krylov solver did not converge in %i iterations (PETSc
    reason %s,
    residual norm ||r|| = %e).",
              num_iterations, reason_str, rnorm);
    }
    */
  }

  // Report results
  // if (report && dolfinx::MPI::rank(this->mpi_comm()) == 0)
  //  write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_dm(DM dm)
{
  assert(_ksp);
  KSPSetDM(_ksp, dm);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_dm_active(bool val)
{
  assert(_ksp);
  if (val)
    KSPSetDMActive(_ksp, PETSC_TRUE);
  else
    KSPSetDMActive(_ksp, PETSC_FALSE);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix
  assert(_ksp);
  PetscErrorCode ierr = KSPSetOptionsPrefix(_ksp, options_prefix.c_str());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScKrylovSolver::get_options_prefix() const
{
  assert(_ksp);
  const char* prefix = nullptr;
  PetscErrorCode ierr = KSPGetOptionsPrefix(_ksp, &prefix);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_from_options() const
{
  assert(_ksp);
  PetscErrorCode ierr = KSPSetFromOptions(_ksp);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPSetFromOptions");
}
//-----------------------------------------------------------------------------
KSP PETScKrylovSolver::ksp() const { return _ksp; }
//-----------------------------------------------------------------------------
