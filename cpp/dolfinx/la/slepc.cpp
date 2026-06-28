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

using namespace dolfinx;
using namespace dolfinx::la;

namespace
{
//-----------------------------------------------------------------------------
void check_petsc_error(PetscErrorCode ierr, const char* petsc_function)
{
  if (ierr != 0)
    petsc::error(ierr, __FILE__, petsc_function);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm)
{
  check_petsc_error(EPSCreate(comm, &_eps), "EPSCreate");
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
    check_petsc_error(ierr, "PetscObjectReference");
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
    (void)EPSDestroy(&_eps);
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
  check_petsc_error(EPSSetOperators(_eps, A, B), "EPSSetOperators");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  // Get operators
  Mat A, B;
  assert(_eps);
  check_petsc_error(EPSGetOperators(_eps, &A, &B), "EPSGetOperators");

  PetscInt m(0), n(0);
  check_petsc_error(MatGetSize(A, &m, &n), "MatGetSize");
  solve(m);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::int64_t n)
{
#ifndef NDEBUG
  // Get operators
  Mat A, B;
  assert(_eps);
  check_petsc_error(EPSGetOperators(_eps, &A, &B), "EPSGetOperators");

  PetscInt _m(0), _n(0);
  check_petsc_error(MatGetSize(A, &_m, &_n), "MatGetSize");
  assert(n <= _n);
#endif

  // Set number of eigenpairs to compute
  assert(_eps);
  check_petsc_error(EPSSetDimensions(_eps, n, PETSC_DECIDE, PETSC_DECIDE),
                    "EPSSetDimensions");

  // Set any options from the PETSc database
  check_petsc_error(EPSSetFromOptions(_eps), "EPSSetFromOptions");

  // Solve eigenvalue problem
  check_petsc_error(EPSSolve(_eps), "EPSSolve");

  // Check for convergence
  EPSConvergedReason reason;
  check_petsc_error(EPSGetConvergedReason(_eps, &reason),
                    "EPSGetConvergedReason");
  if (reason < 0)
    spdlog::warn("Eigenvalue solver did not converge");

  // Report solver status
  PetscInt num_iterations = 0;
  check_petsc_error(EPSGetIterationNumber(_eps, &num_iterations),
                    "EPSGetIterationNumber");

  EPSType eps_type = nullptr;
  check_petsc_error(EPSGetType(_eps, &eps_type), "EPSGetType");
  spdlog::info("Eigenvalue solver ({}) converged in {} iterations.", eps_type,
               num_iterations);
}
//-----------------------------------------------------------------------------
std::complex<PetscReal> SLEPcEigenSolver::get_eigenvalue(int i) const
{
  assert(_eps);

  // Get number of computed values
  PetscInt num_computed_eigenvalues;
  check_petsc_error(EPSGetConverged(_eps, &num_computed_eigenvalues),
                    "EPSGetConverged");

  if (i < num_computed_eigenvalues)
  {
#ifdef PETSC_USE_COMPLEX
    PetscScalar l;
    check_petsc_error(EPSGetEigenvalue(_eps, i, &l, nullptr),
                      "EPSGetEigenvalue");
    return l;
#else
    PetscScalar lr, li;
    check_petsc_error(EPSGetEigenvalue(_eps, i, &lr, &li),
                      "EPSGetEigenvalue");
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

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  check_petsc_error(EPSGetConverged(_eps, &num_computed_eigenvalues),
                    "EPSGetConverged");
  if (ii < num_computed_eigenvalues)
    check_petsc_error(EPSGetEigenpair(_eps, ii, &lr, &lc, r, c),
                      "EPSGetEigenpair");
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
  check_petsc_error(EPSGetConverged(_eps, &num_conv), "EPSGetConverged");
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_options_prefix(const std::string& options_prefix)
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetOptionsPrefix(_eps, options_prefix.c_str());
  check_petsc_error(ierr, "EPSSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string SLEPcEigenSolver::get_options_prefix() const
{
  assert(_eps);
  const char* prefix = nullptr;
  PetscErrorCode ierr = EPSGetOptionsPrefix(_eps, &prefix);
  check_petsc_error(ierr, "EPSGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_from_options() const
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetFromOptions(_eps);
  check_petsc_error(ierr, "EPSSetFromOptions");
}
//-----------------------------------------------------------------------------
int SLEPcEigenSolver::get_iteration_number() const
{
  assert(_eps);
  PetscInt num_iter;
  check_petsc_error(EPSGetIterationNumber(_eps, &num_iter),
                    "EPSGetIterationNumber");
  return num_iter;
}
//-----------------------------------------------------------------------------
EPS SLEPcEigenSolver::eps() const { return _eps; }
//-----------------------------------------------------------------------------
MPI_Comm SLEPcEigenSolver::comm() const
{
  assert(_eps);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  check_petsc_error(PetscObjectGetComm((PetscObject)_eps, &mpi_comm),
                    "PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------

#endif
