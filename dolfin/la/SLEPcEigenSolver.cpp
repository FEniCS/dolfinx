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
// Modified by Ola Skavhaug 2008
// Modified by Anders Logg 2008-2012
// Modified by Marie Rognes 2009
// Modified by Fredrik Valdmanis 2011
//
// First added:  2005-08-31
// Last changed: 2012-02-22

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
  dolfin_assert(A.size(0) == A.size(1));

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
  dolfin_assert(A.size(0) == A.size(1));
  dolfin_assert(B.size(0) == A.size(0));
  dolfin_assert(B.size(1) == A.size(1));

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
  dolfin_assert(A->size(0) == A->size(1));

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
  dolfin_assert(A->size(0) == A->size(1));
  dolfin_assert(B->size(0) == A->size(0));
  dolfin_assert(B->size(1) == A->size(1));

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
    EPSDestroy(&eps);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve()
{
  solve(A->size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(std::size_t n)
{
  dolfin_assert(A);

  // Associate matrix (matrices) with eigenvalue solver
  dolfin_assert(A->size(0) == A->size(1));
  if (B)
  {
    dolfin_assert(B->size(0) == B->size(1) && B->size(0) == A->size(0));
    EPSSetOperators(eps, *A->mat(), *B->mat());
  }
  else
    EPSSetOperators(eps, *A->mat(), PETSC_NULL);

  // Set number of eigenpairs to compute
  dolfin_assert(n <= A->size(0));
  EPSSetDimensions(eps, n, PETSC_DECIDE, PETSC_DECIDE);

  // Set parameters from local parameters
  read_parameters();

  // Set parameters from PETSc parameter database
  EPSSetFromOptions(eps);

  if (parameters["verbose"])
  {
    KSP ksp;
    ST st;
    EPSMonitorSet(eps, EPSMonitorAll, PETSC_NULL, PETSC_NULL);
    EPSSetType(eps, EPSARPACK);
    EPSGetST(eps, &st);
    STGetKSP(st, &ksp);
    KSPMonitorSet(ksp, KSPMonitorDefault, PETSC_NULL, PETSC_NULL);
    EPSView(eps, PETSC_VIEWER_STDOUT_SELF);
  }


  // Solve
  EPSSolve(eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(eps, &reason);
  if (reason < 0)
    warning("Eigenvalue solver did not converge");

  // Report solver status
  dolfin::la_index num_iterations = 0;
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
void SLEPcEigenSolver::get_eigenvalue(double& lr, double& lc, std::size_t i) const
{
  const dolfin::la_index ii = static_cast<dolfin::la_index>(i);

  // Get number of computed values
  dolfin::la_index num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
    EPSGetEigenvalue(eps, ii, &lr, &lc);
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
  const dolfin::la_index ii = static_cast<dolfin::la_index>(i);

  // Get number of computed eigenvectors/values
  dolfin::la_index num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
  {
    dolfin_assert(A);
    A->resize(r, 0);
    A->resize(c, 0);

    dolfin_assert(r.vec());
    dolfin_assert(c.vec());
    EPSGetEigenpair(eps, ii, &lr, &lc, *r.vec(), *c.vec());
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
  dolfin::la_index num_conv;
  EPSGetConverged(eps, &num_conv);
  return num_conv;
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_deflation_space(const PETScVector& deflation_space)
{
  EPSSetDeflationSpace(eps, 1, deflation_space.vec().get());
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
  else if (type == "pos_gen_non_hermitian")
    EPSSetProblemType(eps, EPS_PGNHEP);
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

  ST st;
  EPSGetST(eps, &st);
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
  else if (solver == "arpack")
    EPSSetType(eps, EPSARPACK);
  else
  {
    dolfin_error("SLEPcEigenSolver.cpp",
                 "set solver for SLEPc eigensolver",
                 "Unknown solver type (\"%s\")", solver.c_str());
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::set_tolerance(double tolerance, std::size_t maxiter)
{
  dolfin_assert(tolerance > 0.0);
  EPSSetTolerances(eps, tolerance, maxiter);
}
//-----------------------------------------------------------------------------
std::size_t SLEPcEigenSolver::get_iteration_number() const
{
  dolfin::la_index num_iter;
  EPSGetIterationNumber(eps, &num_iter);
  return num_iter;
}
//-----------------------------------------------------------------------------

#endif
