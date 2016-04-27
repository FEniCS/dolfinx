// Copyright (C) 2005-2014 Garth N. Wells
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
// Modified by Ola Skavhaug 2008
// Modified by Anders Logg 2008-2012
// Modified by Marie Rognes 2009
// Modified by Fredrik Valdmanis 2011

#ifdef HAS_SLEPC

#include <slepcversion.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SLEPcEigenSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm)
{
  // Set up solver environment
  EPSCreate(comm, &_eps);

  // Set default parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(EPS eps) : _eps(eps)
{
  PetscErrorCode ierr;
  if (_eps)
  {
    // Increment reference count since we holding a pointer to it
    ierr = PetscObjectReference((PetscObject)_eps);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PetscObjectReference");
  }
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "intialise SLEPcEigenSolver with SLEPc EPS object",
                 "SLEPc EPS must be initialised (EPSCreate) before wrapping");
  }

  // Set default parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A)
  : SLEPcEigenSolver(A,  nullptr)
{
  // TODO: deprecate

  // Do nothing (handled by other constructor)
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm,
                                   std::shared_ptr<const PETScMatrix> A)
  : SLEPcEigenSolver(comm, A, nullptr)
{
  // TODO: deprecate

  // Do nothing (handled by other constructor)
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(std::shared_ptr<const PETScMatrix> A,
                                   std::shared_ptr<const PETScMatrix> B)
  : _matA(A), _matB(B), _eps(nullptr)

{
  // TODO: deprecate

  dolfin_assert(A);
  dolfin_assert(A->size(0) == A->size(1));
  if (B)
  {
    dolfin_assert(B->size(0) == A->size(0));
    dolfin_assert(B->size(1) == A->size(1));
  }

  // Set up solver environment
  EPSCreate(A->mpi_comm(), &_eps);

  // Set operators
  dolfin_assert(_eps);
  if (B)
    EPSSetOperators(_eps, _matA->mat(), _matB->mat());
  else
    EPSSetOperators(_eps, _matA->mat(), NULL);

  // Set default parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(MPI_Comm comm,
                                   std::shared_ptr<const PETScMatrix> A,
                                   std::shared_ptr<const PETScMatrix> B)
  : _matA(A), _matB(B), _eps(nullptr)
{
  // TODO: deprecate

  dolfin_assert(A);
  dolfin_assert(A->size(0) == A->size(1));
  if (B)
  {
    dolfin_assert(B->size(0) == A->size(0));
    dolfin_assert(B->size(1) == A->size(1));
  }

  // Set default parameter values
  parameters = default_parameters();

  // Set up solver environment
  EPSCreate(comm, &_eps);

  // Set operators
  if (B)
    EPSSetOperators(_eps, _matA->mat(), _matB->mat());
  else
    EPSSetOperators(_eps, _matA->mat(), NULL);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::~SLEPcEigenSolver()
{
  // Destroy solver environment
  if (_eps)
    EPSDestroy(&_eps);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_operators(std::shared_ptr<const PETScMatrix> A,
                                     std::shared_ptr<const PETScMatrix> B)
{
  // Set operators
  dolfin_assert(_eps);
  if (B)
     EPSSetOperators(_eps, _matA->mat(), _matB->mat());
  else
    EPSSetOperators(_eps, _matA->mat(), NULL);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  // Get operators
  Mat A, B;
  dolfin_assert(_eps);
  EPSGetOperators(_eps, &A, &B);

  // Wrap operator as short-cut to get size
  PETScMatrix A_wrapped(A);
  solve(A_wrapped.size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::size_t n)
{
  // Set number of eigenpairs to compute
  dolfin_assert(_matA);
  dolfin_assert(n <= _matA->size(0));
  dolfin_assert(_eps);
  EPSSetDimensions(_eps, n, PETSC_DECIDE, PETSC_DECIDE);

  // Set parameters set on SLEPcEigenSolver object
  read_parameters();

  // Set any options from the PETSc database
  EPSSetFromOptions(_eps);

  // FIXME: need to be able to turn the monitor off
  if (parameters["verbose"].is_set())
  {
    if (parameters["verbose"])
    {
      #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 6 && PETSC_VERSION_RELEASE == 1
      KSP ksp;
      ST st;
      EPSMonitorSet(_eps, EPSMonitorAll,
                    PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)_eps)),
                    NULL);
      EPSGetST(_eps, &st);
      STGetKSP(st, &ksp);
      KSPMonitorSet(ksp, KSPMonitorDefault,
                    PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)),
                    NULL);
      EPSView(_eps, PETSC_VIEWER_STDOUT_SELF);
      #else
      PetscViewerAndFormat *vf;
      PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf);
      EPSMonitorSet(_eps,(PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,
                                             PetscReal*,PetscInt,void*))EPSMonitorAll,vf,
                    (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
      #endif
    }
  }

  // Solve eigenvalue problem
  EPSSolve(_eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(_eps, &reason);
  if (reason < 0)
    warning("Eigenvalue solver did not converge");

  // Report solver status
  PetscInt num_iterations = 0;
  EPSGetIterationNumber(_eps, &num_iterations);

  EPSType eps_type = NULL;
  EPSGetType(_eps, &eps_type);
  log(PROGRESS, "Eigenvalue solver (%s) converged in %d iterations.",
      eps_type, num_iterations);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenvalue(double& lr, double& lc) const
{
  get_eigenvalue(lr, lc, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                     GenericVector& r, GenericVector& c) const
{
  PETScVector& _r = as_type<PETScVector>(r);
  PETScVector& _c = as_type<PETScVector>(c);
  get_eigenpair(lr, lc, _r, _c, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                     PETScVector& r, PETScVector& c) const
{
  get_eigenpair(lr, lc, r, c, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenvalue(double& lr, double& lc,
                                      std::size_t i) const
{
  dolfin_assert(_eps);
  const PetscInt ii = static_cast<PetscInt>(i);

  // Get number of computed values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(_eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
    EPSGetEigenvalue(_eps, ii, &lr, &lc);
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "extract eigenvalue from SLEPc eigenvalue solver",
                 "Requested eigenvalue (%d) has not been computed", i);
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                     GenericVector& r, GenericVector& c,
                                     std::size_t i) const
{
  PETScVector& _r = as_type<PETScVector>(r);
  PETScVector& _c = as_type<PETScVector>(c);
  get_eigenpair(lr, lc, _r, _c, i);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                     PETScVector& r, PETScVector& c,
                                     std::size_t i) const
{
  dolfin_assert(_eps);
  const PetscInt ii = static_cast<PetscInt>(i);

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(_eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    dolfin_assert(_matA);
    _matA->init_vector(r, 0);
    _matA->init_vector(c, 0);

    // Get eigen pairs
    EPSGetEigenpair(_eps, ii, &lr, &lc, r.vec(), c.vec());
  }
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "extract eigenpair from SLEPc eigenvalue solver",
                 "Requested eigenpair (%d) has not been computed", i);
  }
}
//-----------------------------------------------------------------------------
std::size_t SLEPcEigenSolver::get_number_converged() const
{
  PetscInt num_conv;
  dolfin_assert(_eps);
  EPSGetConverged(_eps, &num_conv);
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_deflation_space(const PETScVector& deflation_space)
{
  dolfin_assert(_eps);
  dolfin_assert(deflation_space.vec());
  Vec x = deflation_space.vec();
  dolfin_assert(_eps);
  EPSSetDeflationSpace(_eps, 1, &x);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix
  dolfin_assert(_eps);
  PetscErrorCode ierr = EPSSetOptionsPrefix(_eps, options_prefix.c_str());
  if (ierr != 0) petsc_error(ierr, __FILE__, "EPSSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string SLEPcEigenSolver::get_options_prefix() const
{
  dolfin_assert(_eps);
  const char* prefix = NULL;
  PetscErrorCode ierr = EPSGetOptionsPrefix(_eps, &prefix);
  if (ierr != 0) petsc_error(ierr, __FILE__, "EPSGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::read_parameters()
{
  if (parameters["problem_type"].is_set())
    set_problem_type(parameters["problem_type"]);

  if (parameters["spectrum"].is_set())
    set_spectrum(parameters["spectrum"]);

  if (parameters["solver"].is_set())
    set_solver(parameters["solver"]);

  if (parameters["tolerance"].is_set() or parameters["maximum_iterations"].is_set())
  {
    const double tol = parameters["tolerance"].is_set() ? (double)parameters["tolerance"] : PETSC_DEFAULT;
    const int max_it  = parameters["maximum_iterations"].is_set() ? (int)parameters["maximum_iterations"] : PETSC_DEFAULT;

    set_tolerance(tol, max_it);
  }

  if (parameters["spectral_transform"].is_set())
  {
    if (parameters["spectral_shift"].is_set())
    {
      set_spectral_transform(parameters["spectral_transform"],
                             parameters["spectral_shift"]);
    }
    else
    {
      dolfin_error("SLEPcEigenSolver.cpp",
                   "set spectral transform",
                   "For an spectral transform, the spectral shift parameter must be set");
    }
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_problem_type(std::string type)
{
  // Do nothing if default type is specified
  if (type == "default")
    return;

  dolfin_assert(_eps);
  if (type == "hermitian")
    EPSSetProblemType(_eps, EPS_HEP);
  else if (type == "non_hermitian")
    EPSSetProblemType(_eps, EPS_NHEP);
  else if (type == "gen_hermitian")
    EPSSetProblemType(_eps, EPS_GHEP);
  else if (type == "gen_non_hermitian")
    EPSSetProblemType(_eps, EPS_GNHEP);
  else if (type == "pos_gen_non_hermitian")
    EPSSetProblemType(_eps, EPS_PGNHEP);
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "set problem type for SLEPc eigensolver",
                 "Unknown problem type (\"%s\")", type.c_str());
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_spectral_transform(std::string transform,
                                              double shift)
{
  if (transform == "default")
    return;

  dolfin_assert(_eps);
  ST st;
  EPSGetST(_eps, &st);
  if (transform == "shift-and-invert")
  {
    STSetType(st, STSINVERT);
    STSetShift(st, shift);
  }
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "set spectral transform for SLEPc eigensolver",
                 "Unknown transform (\"%s\")", transform.c_str());
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_spectrum(std::string spectrum)
{
  // Do nothing if default type is specified
  if (spectrum == "default")
    return;

  // Choose spectrum
  dolfin_assert(_eps);
  if (spectrum == "largest magnitude")
    EPSSetWhichEigenpairs(_eps, EPS_LARGEST_MAGNITUDE);
  else if (spectrum == "smallest magnitude")
    EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_MAGNITUDE);
  else if (spectrum == "largest real")
    EPSSetWhichEigenpairs(_eps, EPS_LARGEST_REAL);
  else if (spectrum == "smallest real")
    EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_REAL);
  else if (spectrum == "largest imaginary")
    EPSSetWhichEigenpairs(_eps, EPS_LARGEST_IMAGINARY);
  else if (spectrum == "smallest imaginary")
    EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_IMAGINARY);
  else if (spectrum == "target magnitude")
  {
    EPSSetWhichEigenpairs(_eps, EPS_TARGET_MAGNITUDE);
    if (parameters["spectral_shift"].is_set())
      EPSSetTarget(_eps, parameters["spectral_shift"]);
  }
  else if (spectrum == "target real")
  {
    EPSSetWhichEigenpairs(_eps, EPS_TARGET_REAL);
    if (parameters["spectral_shift"].is_set())
      EPSSetTarget(_eps, parameters["spectral_shift"]);
  }
  else if (spectrum == "target imaginary")
  {
    EPSSetWhichEigenpairs(_eps, EPS_TARGET_IMAGINARY);
    if (parameters["spectral_shift"].is_set())
      EPSSetTarget(_eps, parameters["spectral_shift"]);
  }
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "set spectrum for SLEPc eigensolver",
                 "Unknown spectrum type (\"%s\")", spectrum.c_str());
  }

  // FIXME: Need to add some test here as most algorithms only compute
  // FIXME: largest eigenvalues. Asking for smallest leads to a PETSc error.
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_solver(std::string solver)
{
  // Do nothing if default type is specified
  if (solver == "default")
    return;

  // Choose solver (Note that lanczos will give PETSc error unless
  // problem_type is set to 'hermitian' or 'gen_hermitian')
  dolfin_assert(_eps);
  if (solver == "power")
    EPSSetType(_eps, EPSPOWER);
  else if (solver == "subspace")
    EPSSetType(_eps, EPSSUBSPACE);
  else if (solver == "arnoldi")
    EPSSetType(_eps, EPSARNOLDI);
  else if (solver == "lanczos")
    EPSSetType(_eps, EPSLANCZOS);
  else if (solver == "krylov-schur")
    EPSSetType(_eps, EPSKRYLOVSCHUR);
  else if (solver == "lapack")
    EPSSetType(_eps, EPSLAPACK);
  else if (solver == "arpack")
    EPSSetType(_eps, EPSARPACK);
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "set solver for SLEPc eigensolver",
                 "Unknown solver type (\"%s\")", solver.c_str());
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_tolerance(double tolerance, int maxiter)
{
  dolfin_assert(tolerance > 0.0);
  dolfin_assert(_eps);
  EPSSetTolerances(_eps, tolerance, maxiter);
}
//-----------------------------------------------------------------------------
std::size_t SLEPcEigenSolver::get_iteration_number() const
{
  dolfin_assert(_eps);
  PetscInt num_iter;
  EPSGetIterationNumber(_eps, &num_iter);
  return num_iter;
}
//-----------------------------------------------------------------------------
EPS SLEPcEigenSolver::eps() const
{
  return _eps;
}
//-----------------------------------------------------------------------------

#endif
