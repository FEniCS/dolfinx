// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SLEPC

#include "slepc.h"
#include "petsc.h"
#include <cassert>
#include <dolfinx/common/log.h>
#include <format>
#include <stdexcept>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc::error(ierr, __FILE__, NAME);                                      \
  } while (0)

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm) : _eps(nullptr)
{
  PetscErrorCode ierr = EPSCreate(comm, &_eps);
  CHECK_ERROR("EPSCreate");
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(EPS eps, bool inc_ref_count) : _eps(eps)
{
  if (!eps)
    throw std::runtime_error("SLEPc EPS must be initialised before wrapping");

  if (inc_ref_count)
  {
    PetscErrorCode ierr = PetscObjectReference((PetscObject)_eps);
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
  assert(A);
  assert(_eps);
  PetscErrorCode ierr = EPSSetOperators(_eps, A, B);
  CHECK_ERROR("EPSSetOperators");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  // Get operators
  assert(_eps);
  Mat A = nullptr;
  PetscErrorCode ierr = EPSGetOperators(_eps, &A, nullptr);
  CHECK_ERROR("EPSGetOperators");
  if (!A)
    throw std::runtime_error("Operators must be set before calling solve");

  PetscInt m(0), n(0);
  ierr = MatGetSize(A, &m, &n);
  CHECK_ERROR("MatGetSize");
  solve(n);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::int64_t n)
{
  if (n <= 0)
    throw std::runtime_error("Number of requested eigenpairs must be > 0");

  assert(_eps);
  PetscErrorCode ierr;

#ifndef NDEBUG
  // Get operators
  Mat A = nullptr;
  ierr = EPSGetOperators(_eps, &A, nullptr);
  CHECK_ERROR("EPSGetOperators");
  if (!A)
    throw std::runtime_error("Operators must be set before calling solve");

  PetscInt _m(0), _n(0);
  ierr = MatGetSize(A, &_m, &_n);
  CHECK_ERROR("MatGetSize");
  assert(n <= _n);
#endif

  // Set number of eigenpairs to compute
  ierr = EPSSetDimensions(_eps, static_cast<PetscInt>(n), PETSC_DECIDE,
                          PETSC_DECIDE);
  CHECK_ERROR("EPSSetDimensions");

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
  spdlog::info("Eigenvalue solver ({}) converged in {} iterations.",
               eps_type ? eps_type : "unknown", num_iterations);
}
//-----------------------------------------------------------------------------
std::complex<PetscReal> SLEPcEigenSolver::get_eigenvalue(std::int64_t i) const
{
  assert(_eps);
  if (i < 0)
    throw std::runtime_error("Requested eigenvalue index cannot be negative");

  // Get number of computed values
  PetscInt num_computed_eigenvalues;
  PetscErrorCode ierr = EPSGetConverged(_eps, &num_computed_eigenvalues);
  CHECK_ERROR("EPSGetConverged");

  if (i >= num_computed_eigenvalues)
  {
    throw std::runtime_error(
        std::format("Requested eigenvalue ({}) has not been computed", i));
  }

  const PetscInt ii = static_cast<PetscInt>(i);
#ifdef PETSC_USE_COMPLEX
  PetscScalar l;
  ierr = EPSGetEigenvalue(_eps, ii, &l, nullptr);
  CHECK_ERROR("EPSGetEigenvalue");
  return l;
#else
  PetscScalar lr, li;
  ierr = EPSGetEigenvalue(_eps, ii, &lr, &li);
  CHECK_ERROR("EPSGetEigenvalue");
  return std::complex<PetscReal>(lr, li);
#endif
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(PetscScalar& lr, PetscScalar& lc, Vec r,
                                     Vec c, std::int64_t i) const
{
  assert(_eps);
  if (i < 0)
    throw std::runtime_error("Requested eigenpair index cannot be negative");

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  PetscErrorCode ierr = EPSGetConverged(_eps, &num_computed_eigenvalues);
  CHECK_ERROR("EPSGetConverged");

  if (i >= num_computed_eigenvalues)
  {
    throw std::runtime_error(
        std::format("Requested eigenpair ({}) has not been computed", i));
  }

  ierr = EPSGetEigenpair(_eps, static_cast<PetscInt>(i), &lr, &lc, r, c);
  CHECK_ERROR("EPSGetEigenpair");
}
//-----------------------------------------------------------------------------
std::int64_t SLEPcEigenSolver::get_number_converged() const
{
  assert(_eps);
  PetscInt num_conv;
  PetscErrorCode ierr = EPSGetConverged(_eps, &num_conv);
  CHECK_ERROR("EPSGetConverged");
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_options_prefix(std::string_view options_prefix)
{
  assert(_eps);
  PetscErrorCode ierr
      = EPSSetOptionsPrefix(_eps, std::string(options_prefix).c_str());
  CHECK_ERROR("EPSSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string SLEPcEigenSolver::get_options_prefix() const
{
  assert(_eps);
  const char* prefix = nullptr;
  PetscErrorCode ierr = EPSGetOptionsPrefix(_eps, &prefix);
  CHECK_ERROR("EPSGetOptionsPrefix");
  return prefix ? std::string(prefix) : std::string();
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
