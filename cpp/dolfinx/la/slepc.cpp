// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SLEPC

#include "slepc.h"
#include "petsc.h"
#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <petscmat.h>
#include <slepcversion.h>

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc::error(ierr, __FILE__, NAME);                                      \
  } while (0)

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm)
{
  PetscErrorCode ierr = EPSCreate(comm, &_eps);
  CHECK_ERROR("EPSCreate");
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(EPS eps, bool inc_ref_count) : _eps(eps)
{
  if (!eps)
    throw std::runtime_error("SLEPc EPS must be initialised before wrapping");

  PetscErrorCode ierr;
  if (inc_ref_count)
  {
    ierr = PetscObjectReference((PetscObject)_eps);
    CHECK_ERROR("PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(SLEPcEigenSolver&& solver) noexcept
    : _eps(std::exchange(solver._eps, nullptr))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::~SLEPcEigenSolver()
{
  if (_eps)
    EPSDestroy(&_eps);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver&
SLEPcEigenSolver::operator=(SLEPcEigenSolver&& solver) noexcept
{
  std::swap(_eps, solver._eps);
  return *this;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_operators(const Mat A, const Mat B)
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetOperators(_eps, A, B);
  CHECK_ERROR("EPSSetOperators");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  // Get operators
  Mat A, B;
  assert(_eps);
  PetscErrorCode ierr = EPSGetOperators(_eps, &A, &B);
  CHECK_ERROR("EPSGetOperators");

  PetscInt m(0), n(0);
  ierr = MatGetSize(A, &m, &n);
  CHECK_ERROR("MatGetSize");

  solve(m);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::int64_t n)
{
  PetscErrorCode ierr;
#ifndef NDEBUG
  // Get operators
  Mat A, B;
  assert(_eps);
  ierr = EPSGetOperators(_eps, &A, &B);
  CHECK_ERROR("EPSGetOperators");

  PetscInt _m(0), _n(0);
  ierr = MatGetSize(A, &_m, &_n);
  CHECK_ERROR("MatGetSize");

  assert(n <= _n);
#endif

  // Set number of eigenpairs to compute
  assert(_eps);

  ierr = EPSSetDimensions(_eps, n, PETSC_DECIDE, PETSC_DECIDE);
  CHECK_ERROR("EPSSetDimensions");

  // Set any options from the PETSc database
  ierr = EPSSetFromOptions(_eps);
  CHECK_ERROR("EPSSetFromOptions");

  // Solve eigenvalue problem
  ierr = EPSSolve(_eps);
  CHECK_ERROR("EPSSolve");

  // Check for convergence
  EPSConvergedReason reason;
  ierr = EPSGetConvergedReason(_eps, &reason);
  CHECK_ERROR("EPSGetConvergedReason");
  if (reason < 0)
    spdlog::warn("Eigenvalue solver did not converge");

  // Report solver status
  PetscInt num_iterations = 0;
  ierr = EPSGetIterationNumber(_eps, &num_iterations);
  CHECK_ERROR("EPSGetIterationNumber");

  EPSType eps_type = nullptr;
  ierr = EPSGetType(_eps, &eps_type);
  CHECK_ERROR("EPSGetType");
  spdlog::info("Eigenvalue solver ({}) converged in {} iterations.", eps_type,
               num_iterations);
}
//-----------------------------------------------------------------------------
std::complex<PetscReal> SLEPcEigenSolver::get_eigenvalue(int i) const
{
  assert(_eps);
  PetscErrorCode ierr;
  // Get number of computed values
  PetscInt num_computed_eigenvalues;
  ierr = EPSGetConverged(_eps, &num_computed_eigenvalues);
  CHECK_ERROR("EPSGetConverged");

  if (i < num_computed_eigenvalues)
  {
#ifdef PETSC_USE_COMPLEX
    PetscScalar l;
    ierr = EPSGetEigenvalue(_eps, i, &l, nullptr);
    CHECK_ERROR("EPSGetEigenvalue");
    return l;
#else
    PetscScalar lr, li;
    ierr = EPSGetEigenvalue(_eps, i, &lr, &li);
    CHECK_ERROR("EPSGetEigenvalue");
    return std::complex<PetscReal>(lr, li);
#endif
  }
  else
  {
    throw std::runtime_error("Requested eigenvalue (" + std::to_string(i)
                             + ") has not been computed");
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(PetscScalar& lr, PetscScalar& lc, Vec r,
                                     Vec c, int i) const
{
  assert(_eps);
  PetscInt ii = static_cast<PetscInt>(i);
  PetscErrorCode ierr;

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  ierr = EPSGetConverged(_eps, &num_computed_eigenvalues);
  CHECK_ERROR("EPSGetConverged");
  if (ii < num_computed_eigenvalues)
  {
    ierr = EPSGetEigenpair(_eps, ii, &lr, &lc, r, c);
    CHECK_ERROR("EPSGetEigenpair");
  }
  else
  {
    throw std::runtime_error("Requested eigenpair (" + std::to_string(i)
                             + ") has not been computed");
  }
}
//-----------------------------------------------------------------------------
std::int64_t SLEPcEigenSolver::get_number_converged() const
{
  PetscInt num_conv;
  assert(_eps);
  PetscErrorCode ierr = EPSGetConverged(_eps, &num_conv);
  CHECK_ERROR("EPSGetConverged");
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_options_prefix(const std::string& options_prefix)
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetOptionsPrefix(_eps, options_prefix.c_str());
  CHECK_ERROR("EPSSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string SLEPcEigenSolver::get_options_prefix() const
{
  assert(_eps);
  const char* prefix = nullptr;
  PetscErrorCode ierr = EPSGetOptionsPrefix(_eps, &prefix);
  CHECK_ERROR("EPSGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_from_options() const
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetFromOptions(_eps);
  CHECK_ERROR("EPSSetFromOptions");
}
//-----------------------------------------------------------------------------
int SLEPcEigenSolver::get_iteration_number() const
{
  assert(_eps);
  PetscInt num_iter;
  PetscErrorCode ierr = EPSGetIterationNumber(_eps, &num_iter);
  CHECK_ERROR("EPSGetIterationNumber");
  return num_iter;
}
//-----------------------------------------------------------------------------
EPS SLEPcEigenSolver::eps() const { return _eps; }
//-----------------------------------------------------------------------------
MPI_Comm SLEPcEigenSolver::comm() const
{
  assert(_eps);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)_eps, &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------

#endif