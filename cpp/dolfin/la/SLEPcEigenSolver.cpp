// Copyright (C) 2005-2017 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SLEPC

#include "SLEPcEigenSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include <slepcversion.h>

using namespace dolfin;
using namespace dolfin::la;

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
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "PetscObjectReference");
  }
  else
  {
    log::dolfin_error(
        "SLEPcEigenSolver.cpp",
        "initialize SLEPcEigenSolver with SLEPc EPS object",
        "SLEPc EPS must be initialised (EPSCreate) before wrapping");
  }

  // Set default parameter values
  parameters = default_parameters();
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
  assert(_eps);
  if (B)
    EPSSetOperators(_eps, A->mat(), B->mat());
  else
    EPSSetOperators(_eps, A->mat(), NULL);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  // Get operators
  Mat A, B;
  assert(_eps);
  EPSGetOperators(_eps, &A, &B);

  // Wrap operator as short-cut to get size
  PETScMatrix A_wrapped(A);
  solve(A_wrapped.size()[0]);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::int64_t n)
{
#ifdef DEBUG
  // Get operators
  Mat A, B;
  assert(_eps);
  EPSGetOperators(_eps, &A, &B);

  // Wrap operator as short-cut to get size
  PETScMatrix A_wrapped(A);
  assert(n <= A_wrapped.size()[0]);
#endif

  // Set number of eigenpairs to compute
  assert(_eps);
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
      PetscViewerAndFormat* vf;
      PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                 PETSC_VIEWER_DEFAULT, &vf);
      EPSMonitorSet(_eps,
                    (PetscErrorCode(*)(EPS, PetscInt, PetscInt, PetscScalar*,
                                       PetscScalar*, PetscReal*, PetscInt,
                                       void*))EPSMonitorAll,
                    vf, (PetscErrorCode(*)(void**))PetscViewerAndFormatDestroy);
    }
  }

  // Solve eigenvalue problem
  EPSSolve(_eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(_eps, &reason);
  if (reason < 0)
    log::warning("Eigenvalue solver did not converge");

  // Report solver status
  PetscInt num_iterations = 0;
  EPSGetIterationNumber(_eps, &num_iterations);

  EPSType eps_type = NULL;
  EPSGetType(_eps, &eps_type);
  log::log(PROGRESS, "Eigenvalue solver (%s) converged in %d iterations.",
           eps_type, num_iterations);
}
//-----------------------------------------------------------------------------
std::complex<PetscReal> SLEPcEigenSolver::get_eigenvalue(std::size_t i) const
{
  assert(_eps);
  const PetscInt ii = static_cast<PetscInt>(i);

  // Get number of computed values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(_eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
#ifdef PETSC_USE_COMPLEX
    PetscScalar l;
    EPSGetEigenvalue(_eps, ii, &l, NULL);
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
void SLEPcEigenSolver::get_eigenpair(PetscScalar& lr, PetscScalar& lc,
                                     PETScVector& r, PETScVector& c,
                                     std::size_t i) const
{
  assert(_eps);
  const PetscInt ii = static_cast<PetscInt>(i);

  // Get number of computed eigenvectors/values
  PetscInt num_computed_eigenvalues;
  EPSGetConverged(_eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    // Get operators
    Mat A, B;
    assert(_eps);
    EPSGetOperators(_eps, &A, &B);

    // Wrap operator and initialize r and c
    PETScMatrix A_wrapped(A);
    r = A_wrapped.init_vector(0);
    c = A_wrapped.init_vector(0);

    // Get eigen pairs
    EPSGetEigenpair(_eps, ii, &lr, &lc, r.vec(), c.vec());
  }
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
void SLEPcEigenSolver::set_deflation_space(
    const la::VectorSpaceBasis& deflation_space)
{
  assert(_eps);

  // Get PETSc vector pointers from VectorSpaceBasis
  std::vector<Vec> petsc_vecs(deflation_space.dim());
  for (std::size_t i = 0; i < deflation_space.dim(); ++i)
  {
    assert(deflation_space[i]);
    assert(deflation_space[i]->vec());
    petsc_vecs[i] = deflation_space[i]->vec();
  }

  PetscErrorCode ierr
      = EPSSetDeflationSpace(_eps, petsc_vecs.size(), petsc_vecs.data());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "EPSSetDeflationSpace");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_initial_space(
    const la::VectorSpaceBasis& initial_space)
{
  assert(_eps);

  // Get PETSc vector pointers from VectorSpaceBasis
  std::vector<Vec> petsc_vecs(initial_space.dim());
  for (std::size_t i = 0; i < initial_space.dim(); ++i)
  {
    assert(initial_space[i]);
    assert(initial_space[i]->vec());
    petsc_vecs[i] = initial_space[i]->vec();
  }

  PetscErrorCode ierr
      = EPSSetInitialSpace(_eps, petsc_vecs.size(), petsc_vecs.data());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "EPSSetInitialSpace");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix
  assert(_eps);
  PetscErrorCode ierr = EPSSetOptionsPrefix(_eps, options_prefix.c_str());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "EPSSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string SLEPcEigenSolver::get_options_prefix() const
{
  assert(_eps);
  const char* prefix = NULL;
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
void SLEPcEigenSolver::read_parameters()
{
  if (parameters["problem_type"].is_set())
    set_problem_type(parameters["problem_type"]);

  if (parameters["spectrum"].is_set())
    set_spectrum(parameters["spectrum"]);

  if (parameters["solver"].is_set())
    set_solver(parameters["solver"]);

  if (parameters["tolerance"].is_set()
      or parameters["maximum_iterations"].is_set())
  {
    const double tol = parameters["tolerance"].is_set()
                           ? (double)parameters["tolerance"]
                           : PETSC_DEFAULT;
    const int max_it = parameters["maximum_iterations"].is_set()
                           ? (int)parameters["maximum_iterations"]
                           : PETSC_DEFAULT;

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
      throw std::runtime_error("For an spectral transform, the spectral shift "
                               "parameter must be set");
    }
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_problem_type(std::string type)
{
  // Do nothing if default type is specified
  if (type == "default")
    return;

  assert(_eps);
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
    throw std::runtime_error("Unknown problem type (" + type + ")");
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_spectral_transform(std::string transform,
                                              double shift)
{
  if (transform == "default")
    return;

  assert(_eps);
  ST st;
  EPSGetST(_eps, &st);
  if (transform == "shift-and-invert")
  {
    STSetType(st, STSINVERT);
    STSetShift(st, shift);
  }
  else
  {
    throw std::runtime_error("Unknown transform (" + transform + ")");
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_spectrum(std::string spectrum)
{
  // Do nothing if default type is specified
  if (spectrum == "default")
    return;

  // Choose spectrum
  assert(_eps);
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
    {
      PetscScalar shift = (double)parameters["spectral_shift"];
      EPSSetTarget(_eps, shift);
    }
  }
  else if (spectrum == "target real")
  {
    EPSSetWhichEigenpairs(_eps, EPS_TARGET_REAL);
    if (parameters["spectral_shift"].is_set())
    {
      PetscScalar shift = (double)parameters["spectral_shift"];
      EPSSetTarget(_eps, shift);
    }
  }
  else if (spectrum == "target imaginary")
  {
    EPSSetWhichEigenpairs(_eps, EPS_TARGET_IMAGINARY);
    if (parameters["spectral_shift"].is_set())
    {
      PetscScalar shift = (double)parameters["spectral_shift"];
      EPSSetTarget(_eps, shift);
    }
  }
  else
  {
    throw std::runtime_error("Unknown spectrum type (" + spectrum + ")");
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
  assert(_eps);
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
  else if (solver == "jacobi-davidson")
    EPSSetType(_eps, EPSJD);
  else if (solver == "generalized-davidson")
    EPSSetType(_eps, EPSGD);
  else
  {
    throw std::runtime_error("Unknown solver type (" + solver + ")");
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_tolerance(double tolerance, int maxiter)
{
  assert(tolerance > 0.0);
  assert(_eps);
  EPSSetTolerances(_eps, tolerance, maxiter);
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
