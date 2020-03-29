// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SLEPC

#include "SLEPcEigenSolver.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/PETScVector.h>
#include <petscmat.h>
#include <slepcversion.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm) { EPSCreate(comm, &_eps); }
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(EPS eps, bool inc_ref_count) : _eps(eps)
{
  if (!eps)
    throw std::runtime_error("SLEPc EPS must be initialised before wrapping");

  PetscErrorCode ierr;
  if (inc_ref_count)
  {
    ierr = PetscObjectReference((PetscObject)_eps);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::~SLEPcEigenSolver()
{
  if (_eps)
    EPSDestroy(&_eps);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_operators(const Mat A, const Mat B)
{
  // Set operators
  assert(_eps);
  EPSSetOperators(_eps, A, B);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  // Get operators
  Mat A, B;
  assert(_eps);
  EPSGetOperators(_eps, &A, &B);

  PetscInt m(0), n(0);
  MatGetSize(A, &m, &n);
  solve(m);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::int64_t n)
{
#ifdef DEBUG
  // Get operators
  Mat A, B;
  assert(_eps);
  EPSGetOperators(_eps, &A, &B);

  PetscInt _m(0), _n(0);
  MatGetSize(A, &_m, &_n);
  assert(n <= _n);
#endif

  // Set number of eigenpairs to compute
  assert(_eps);
  EPSSetDimensions(_eps, n, PETSC_DECIDE, PETSC_DECIDE);

  // Set any options from the PETSc database
  EPSSetFromOptions(_eps);

  // Solve eigenvalue problem
  EPSSolve(_eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(_eps, &reason);
  if (reason < 0)
    LOG(WARNING) << "Eigenvalue solver did not converge";

  // Report solver status
  PetscInt num_iterations = 0;
  EPSGetIterationNumber(_eps, &num_iterations);

  EPSType eps_type = nullptr;
  EPSGetType(_eps, &eps_type);
  LOG(INFO) << "Eigenvalue solver (" << eps_type << ") converged in "
            << num_iterations << " iterations.";
}
//-----------------------------------------------------------------------------
std::complex<PetscReal> SLEPcEigenSolver::get_eigenvalue(std::size_t i) const
{
  assert(_eps);

  // Get number of computed values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(_eps, &num_computed_eigenvalues);

  const auto ii = static_cast<PetscInt>(i);
  if (ii < num_computed_eigenvalues)
  {
#ifdef PETSC_USE_COMPLEX
    PetscScalar l;
    EPSGetEigenvalue(_eps, ii, &l, nullptr);
    return l;
#else
    PetscScalar lr, li;
    EPSGetEigenvalue(_eps, ii, &lr, &li);
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
                                     Vec c, std::size_t i) const
{
  assert(_eps);
  const auto ii = static_cast<PetscInt>(i);

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(_eps, &num_computed_eigenvalues);
  if (ii < num_computed_eigenvalues)
    EPSGetEigenpair(_eps, ii, &lr, &lc, r, c);
  else
  {
    throw std::runtime_error("Requested eigenpair (" + std::to_string(i)
                             + ") has not been computed");
  }
}
//-----------------------------------------------------------------------------
std::size_t SLEPcEigenSolver::get_number_converged() const
{
  PetscInt num_conv;
  assert(_eps);
  EPSGetConverged(_eps, &num_conv);
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_options_prefix(std::string options_prefix)
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetOptionsPrefix(_eps, options_prefix.c_str());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "EPSSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string SLEPcEigenSolver::get_options_prefix() const
{
  assert(_eps);
  const char* prefix = nullptr;
  PetscErrorCode ierr = EPSGetOptionsPrefix(_eps, &prefix);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "EPSGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_from_options() const
{
  assert(_eps);
  PetscErrorCode ierr = EPSSetFromOptions(_eps);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "EPSSetFromOptions");
}
//-----------------------------------------------------------------------------
std::size_t SLEPcEigenSolver::get_iteration_number() const
{
  assert(_eps);
  PetscInt num_iter;
  EPSGetIterationNumber(_eps, &num_iter);
  return num_iter;
}
//-----------------------------------------------------------------------------
EPS SLEPcEigenSolver::eps() const { return _eps; }
//-----------------------------------------------------------------------------
MPI_Comm SLEPcEigenSolver::mpi_comm() const
{
  assert(_eps);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_eps, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------

#endif
