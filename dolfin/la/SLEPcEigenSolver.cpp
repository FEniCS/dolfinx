// Copyright (C) 2005-2011 Garth N. Wells
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
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008-2009.
// Modified by Marie Rognes, 2009.
//
// First added:  2005-08-31
// Last changed: 2011-03-17

#ifdef HAS_SLEPC

#include <slepcversion.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SLEPcEigenSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(const PETScMatrix& A)
      : A(reference_to_no_delete_pointer(const_cast<PETScMatrix&>(A)))
{
  assert(A.size(0) == A.size(1));

  // Set default parameter values
  parameters = default_parameters();

  // Set up solver environment
  if (dolfin::MPI::num_processes() > 1)
    EPSCreate(PETSC_COMM_WORLD, &eps);
  else
    EPSCreate(PETSC_COMM_SELF, &eps);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(const PETScMatrix& A, const PETScMatrix& B)
   : A(reference_to_no_delete_pointer(A)),
     B(reference_to_no_delete_pointer(B))

{
  assert(A.size(0) == A.size(1));
  assert(B.size(0) == A.size(0));
  assert(B.size(1) == A.size(1));

  // Set default parameter values
  parameters = default_parameters();

  // Set up solver environment
  if (dolfin::MPI::num_processes() > 1)
    EPSCreate(PETSC_COMM_WORLD, &eps);
  else
    EPSCreate(PETSC_COMM_SELF, &eps);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(boost::shared_ptr<const PETScMatrix> A) : A(A)
{
  assert(A->size(0) == A->size(1));

  // Set default parameter values
  parameters = default_parameters();

  // Set up solver environment
  if (dolfin::MPI::num_processes() > 1)
    EPSCreate(PETSC_COMM_WORLD, &eps);
  else
    EPSCreate(PETSC_COMM_SELF, &eps);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver(boost::shared_ptr<const PETScMatrix> A,
                           boost::shared_ptr<const PETScMatrix> B) : A(A), B(B)

{
  assert(A->size(0) == A->size(1));
  assert(B->size(0) == A->size(0));
  assert(B->size(1) == A->size(1));

  // Set default parameter values
  parameters = default_parameters();

  // Set up solver environment
  if (dolfin::MPI::num_processes() > 1)
    EPSCreate(PETSC_COMM_WORLD, &eps);
  else
    EPSCreate(PETSC_COMM_SELF, &eps);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::~SLEPcEigenSolver()
{
  // Destroy solver environment
  if (eps)
    EPSDestroy(eps);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  solve(A->size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(uint n)
{
  assert(A);

  // Associate matrix (matrices) with eigenvalue solver
  assert(A->size(0) == A->size(1));
  if (B)
  {
    assert(B->size(0) == B->size(1) && B->size(0) == A->size(0));
    EPSSetOperators(eps, *A->mat(), *B->mat());
  }
  else
    EPSSetOperators(eps, *A->mat(), PETSC_NULL);

  // Set number of eigenpairs to compute
  assert(n <= A->size(0));
  const uint nn = static_cast<int>(n);
  EPSSetDimensions(eps, nn, PETSC_DECIDE, PETSC_DECIDE);

  // Set parameters from local parameters
  read_parameters();

  // Set parameters from PETSc parameter database
  EPSSetFromOptions(eps);

  // Solve
  EPSSolve(eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(eps, &reason);
  if (reason < 0)
    warning("Eigenvalue solver did not converge");

  // Report solver status
  int num_iterations = 0;
  EPSGetIterationNumber(eps, &num_iterations);

  const EPSType eps_type = 0;
  EPSGetType(eps, &eps_type);
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
                                     PETScVector& r, PETScVector& c) const
{
  get_eigenpair(lr, lc, r, c, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenvalue(double& lr, double& lc, uint i) const
{
  const int ii = static_cast<int>(i);

  // Get number of computed values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    #if SLEPC_VERSION_MAJOR == 3 && SLEPC_VERSION_MINOR == 1
    EPSGetEigenvalue(eps, ii, &lr, &lc);
    #else
    EPSGetValue(eps, ii, &lr, &lc);
    #endif
  }
  else
    error("Requested eigenvalue has not been computed");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::get_eigenpair(double& lr, double& lc,
                                    PETScVector& r, PETScVector& c,
                                    uint i) const
{
  const int ii = static_cast<int>(i);

  // Get number of computed eigenvectors/values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    assert(A);
    A->resize(r, 0);
    A->resize(c, 0);

    assert(r.vec());
    assert(c.vec());
    EPSGetEigenpair(eps, ii, &lr, &lc, *r.vec(), *c.vec());
  }
  else
    error("Requested eigenvalue/vector has not been computed");
}
//-----------------------------------------------------------------------------
int SLEPcEigenSolver::get_number_converged() const
{
  int num_conv;
  EPSGetConverged(eps, &num_conv);
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_deflation_space(const PETScVector& deflation_space)
{
  #if SLEPC_VERSION_MAJOR == 3 && SLEPC_VERSION_MINOR == 1
  EPSSetDeflationSpace(eps, 1, deflation_space.vec().get());
  #else
  error("Setting a deflation space requires SLEPc 3.1 or newer version");
  #endif
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::read_parameters()
{
  set_problem_type(parameters["problem_type"]);
  set_spectrum(parameters["spectrum"]);
  set_solver(parameters["solver"]);
  set_tolerance(parameters["tolerance"], parameters["maximum_iterations"]);
  set_spectral_transform(parameters["spectral_transform"],
                         parameters["spectral_shift"]);

}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_problem_type(std::string type)
{
  // Do nothing if default type is specified
  if (type == "default")
    return;

  if (type == "hermitian")
    EPSSetProblemType(eps, EPS_HEP);
  else if (type == "non_hermitian")
    EPSSetProblemType(eps, EPS_NHEP);
  else if (type == "gen_hermitian")
    EPSSetProblemType(eps, EPS_GHEP);
  else if (type == "gen_non_hermitian")
    EPSSetProblemType(eps, EPS_GNHEP);
  else
    error("Unknown problem type: \"%s\".", type.c_str());
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_spectral_transform(std::string transform,
                                              double shift)
{
  if (transform == "default")
    return;

  ST st;
  EPSGetST(eps, &st);
  if (transform == "shift-and-invert")
  {
    #if SLEPC_VERSION_MAJOR == 3 && SLEPC_VERSION_MINOR == 1
    STSetType(st, STSINVERT);
    #else
    STSetType(st, STSINV);
    #endif
    STSetShift(st, shift);
  }
  else
    error("Unknown transform: \"%s\".", transform.c_str());
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_spectrum(std::string spectrum)
{
  // Do nothing if default type is specified
  if (spectrum == "default")
    return;

  // Choose spectrum
  if (spectrum == "largest magnitude")
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
  else if (spectrum == "smallest magnitude")
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
  else if (spectrum == "largest real")
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL);
  else if (spectrum == "smallest real")
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  else if (spectrum == "largest imaginary")
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_IMAGINARY);
  else if (spectrum == "smallest imaginary")
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_IMAGINARY);

  #if SLEPC_VERSION_MAJOR == 3 && SLEPC_VERSION_MINOR == 1
  else if (spectrum == "target magnitude")
  {
    EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);
    EPSSetTarget(eps, parameters["spectral_shift"]);
  }
  else if (spectrum == "target real")
  {
    EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL);
    EPSSetTarget(eps, parameters["spectral_shift"]);
  }
  else if (spectrum == "target imaginary")
  {
    EPSSetWhichEigenpairs(eps, EPS_TARGET_IMAGINARY);
    EPSSetTarget(eps, parameters["spectral_shift"]);
  }
  #endif

  else
    error("Unknown spectrum: \"%s\".", spectrum.c_str());

  // FIXME: Need to add some test here as most algorithms only compute
  // FIXME: largest eigenvalues. Asking for smallest leads to a PETSc error.
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_solver(std::string solver)
{
  // Do nothing if default type is specified
  if (solver == "default")
    return;

  // Choose solver

  // (Note that lanczos will give PETSc error unless problem_type is
  // set to 'hermitian' or 'gen_hermitian')
  if (solver == "power")
    EPSSetType(eps, EPSPOWER);
  else if (solver == "subspace")
    EPSSetType(eps, EPSSUBSPACE);
  else if (solver == "arnoldi")
    EPSSetType(eps, EPSARNOLDI);
  else if (solver == "lanczos")
    EPSSetType(eps, EPSLANCZOS);
  else if (solver == "krylov-schur")
    EPSSetType(eps, EPSKRYLOVSCHUR);
  else if (solver == "lapack")
    EPSSetType(eps, EPSLAPACK);
  else
    error("Unknown method: \"%s\".", solver.c_str());
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_tolerance(double tolerance, uint maxiter)
{
  assert(tolerance > 0.0);
  EPSSetTolerances(eps, tolerance, static_cast<int>(maxiter));
}
//-----------------------------------------------------------------------------
int SLEPcEigenSolver::get_iteration_number() const
{
  int num_iter;
  EPSGetIterationNumber(eps, &num_iter);
  return num_iter;
}
//-----------------------------------------------------------------------------
#endif
