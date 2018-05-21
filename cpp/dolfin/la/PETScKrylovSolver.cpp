// Copyright (C) 2014 Johan Jansson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScKrylovSolver.h"
#include "PETScMatrix.h"
#include "PETScOperator.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/PETScDMCollection.h>
#include <petsclog.h>

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(MPI_Comm comm) : _ksp(NULL)
{
  PetscErrorCode ierr;

  // Create PETSc KSP object
  ierr = KSPCreate(comm, &_ksp);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPCreate");
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::PETScKrylovSolver(KSP ksp) : _ksp(ksp)
{
  PetscErrorCode ierr;
  if (_ksp)
  {
    // Increment reference count since we holding a pointer to it
    ierr = PetscObjectReference((PetscObject)_ksp);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PetscObjectReference");
  }
  else
  {
    log::dolfin_error(
        "PETScKrylovSolver.cpp",
        "initialize PETScKrylovSolver with PETSc KSP object",
        "PETSc KSP must be initialised (KSPCreate) before wrapping");
  }
}
//-----------------------------------------------------------------------------
PETScKrylovSolver::~PETScKrylovSolver()
{
  // Decrease reference count for KSP object, and clean-up if
  // reference count goes to zero.
  if (_ksp)
    KSPDestroy(&_ksp);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operator(const la::PETScOperator& A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_operators(const la::PETScOperator& A,
                                      const la::PETScOperator& P)
{
  assert(A.mat());
  assert(P.mat());
  assert(_ksp);

  PetscErrorCode ierr;
  ierr = KSPSetOperators(_ksp, A.mat(), P.mat());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::solve(PETScVector& x, const PETScVector& b,
                                     bool transpose)
{
  common::Timer timer("PETSc Krylov solver");

  // Get PETSc operators
  Mat _A, _P;
  KSPGetOperators(_ksp, &_A, &_P);
  assert(_A);

  // Create wrapper around PETSc Mat object
  la::PETScOperator A(_A);

  PetscErrorCode ierr;

  // Check dimensions
  const std::array<std::int64_t, 2> size = A.size();
  if (size[0] != b.size())
  {
    log::dolfin_error(
        "PETScKrylovSolver.cpp",
        "unable to solve linear system with PETSc Krylov solver",
        "Non-matching dimensions for linear system (matrix has %ld "
        "rows and right-hand side vector has %ld rows)",
        size[0], b.size());
  }

  // Initialize solution vector, if necessary
  if (x.empty())
  {
    x = A.init_vector(1);
    // Zero the vector unless PETSc does it for us
    PetscBool nonzero_guess;
    ierr = KSPGetInitialGuessNonzero(_ksp, &nonzero_guess);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPGetInitialGuessNonzero");
    if (nonzero_guess)
      x.set(0.0);
  }

  // Solve linear system
  if (dolfin::MPI::rank(this->mpi_comm()) == 0)
    log::log(PROGRESS, "PETSc Krylov solver starting to solve system.");

  // Solve system
  if (!transpose)
  {
    ierr = KSPSolve(_ksp, b.vec(), x.vec());
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPSolve");
  }
  else
  {
    ierr = KSPSolveTranspose(_ksp, b.vec(), x.vec());
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPSolve");
  }

  // Update ghost values in solution vector
  x.update_ghosts();

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
  // if (report && dolfin::MPI::rank(this->mpi_comm()) == 0)
  //  write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::set_reuse_preconditioner(bool reuse_pc)
{
  assert(_ksp);
  const PetscBool _reuse_pc = reuse_pc ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode ierr = KSPSetReusePreconditioner(_ksp, _reuse_pc);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPSetReusePreconditioner");
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
  const char* prefix = NULL;
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
std::string PETScKrylovSolver::str(bool verbose) const
{
  assert(_ksp);
  std::stringstream s;
  if (verbose)
  {
    log::warning("Verbose output for PETScKrylovSolver not implemented, "
                 "calling PETSc KSPView directly.");
    PetscErrorCode ierr = KSPView(_ksp, PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPView");
  }
  else
    s << "<PETScKrylovSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
MPI_Comm PETScKrylovSolver::mpi_comm() const
{
  assert(_ksp);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_ksp, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
KSP PETScKrylovSolver::ksp() const { return _ksp; }
//-----------------------------------------------------------------------------
std::size_t PETScKrylovSolver::_solve(const la::PETScOperator& A,
                                      PETScVector& x, const PETScVector& b)
{
  // Set operator
  assert(_ksp);
  assert(A.mat());
  KSPSetOperators(_ksp, A.mat(), A.mat());

  // Call solve
  std::size_t num_iter = solve(x, b);

  // Clear operators
  KSPSetOperators(_ksp, nullptr, nullptr);

  return num_iter;
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::write_report(int num_iterations,
                                     KSPConvergedReason reason)
{
  assert(_ksp);

  PetscErrorCode ierr;

  // Get name of solver and preconditioner
  PC pc;
  KSPType ksp_type;
  PCType pc_type;

  ierr = KSPGetType(_ksp, &ksp_type);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPGetType");

  ierr = KSPGetPC(_ksp, &pc);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "KSPGetPC");

  ierr = PCGetType(pc, &pc_type);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "PCGetType");

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
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "PCASMGetSubKSP");
    }
    else if (pc_type_str == PCBJACOBI)
    {
      ierr = PCBJacobiGetSubKSP(pc, NULL, NULL, &sub_ksp);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "PCBJacobiGetSubKSP");
    }
    ierr = KSPGetType(*sub_ksp, &sub_ksp_type);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPGetType");

    ierr = KSPGetPC(*sub_ksp, &sub_pc);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "KSPGetPC");

    ierr = PCGetType(sub_pc, &sub_pc_type);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PCGetType");
  }

  // FIXME: Get preconditioner description from PETScPreconditioner

  // Report number of iterations and solver type
  if (reason >= 0)
  {
    log::log(PROGRESS,
             "PETSc Krylov solver (%s, %s) converged in %d iterations.",
             ksp_type, pc_type, num_iterations);
  }
  else
  {
    log::log(
        PROGRESS,
        "PETSc Krylov solver (%s, %s) failed to converge in %d iterations.",
        ksp_type, pc_type, num_iterations);
  }

  if (pc_type_str == PCASM || pc_type_str == PCBJACOBI)
  {
    log::log(PROGRESS,
             "PETSc Krylov solver preconditioner (%s) submethods: (%s, %s)",
             pc_type, sub_ksp_type, sub_pc_type);
  }

#if PETSC_HAVE_HYPRE
  if (pc_type_str == PCHYPRE)
  {
    const char* hypre_sub_type;
    ierr = PCHYPREGetType(pc, &hypre_sub_type);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PCHYPREGetType");

    log::log(PROGRESS, "  Hypre preconditioner method: %s", hypre_sub_type);
  }
#endif
}
//-----------------------------------------------------------------------------
void PETScKrylovSolver::check_dimensions(const la::PETScOperator& A,
                                         const PETScVector& x,
                                         const PETScVector& b) const
{
  std::array<std::int64_t, 2> size = A.size();

  // Check dimensions of A
  if (size[0] == 0 || size[1] == 0)
  {
    log::dolfin_error(
        "PETScKrylovSolver.cpp",
        "unable to solve linear system with PETSc Krylov solver",
        "Matrix does not have a nonzero number of rows and columns");
  }

  // Check dimensions of A vs b
  if (size[0] != b.size())
  {
    log::dolfin_error(
        "PETScKrylovSolver.cpp",
        "unable to solve linear system with PETSc Krylov solver",
        "Non-matching dimensions for linear system (matrix has %ld "
        "rows and right-hand side vector has %ld rows)",
        size[0], b.size());
  }

  // Check dimensions of A vs x
  if (!x.empty() && x.size() != size[1])
  {
    log::dolfin_error(
        "PETScKrylovSolver.cpp",
        "unable to solve linear system with PETSc Krylov solver",
        "Non-matching dimensions for linear system (matrix has %ld "
        "columns and solution vector has %ld rows)",
        size[1], x.size());
  }
}
//-----------------------------------------------------------------------------
