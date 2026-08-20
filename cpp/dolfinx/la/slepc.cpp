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
  assert(_eps);

  // Solve eigenvalue problem. EPSSetUp errors if no operators have been
  // set
  PetscErrorCode ierr = EPSSolve(_eps);
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
std::complex<PetscReal> SLEPcEigenSolver::get_eigenvalue(PetscInt i) const
{
  assert(_eps);

  // EPSGetEigenvalue checks that the problem has been solved and that i
  // is in range
  PetscErrorCode ierr;
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
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(PetscScalar& lr, PetscScalar& lc, Vec r,
                                     Vec c, PetscInt i) const
{
  assert(_eps);

  // EPSGetEigenpair checks that the problem has been solved and that i
  // is in range
  PetscErrorCode ierr = EPSGetEigenpair(_eps, i, &lr, &lc, r, c);
  CHECK_ERROR("EPSGetEigenpair");
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
