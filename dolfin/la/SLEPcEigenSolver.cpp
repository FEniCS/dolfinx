// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2005-08-31
// Last changed: 2008-09-07

#ifdef HAS_SLEPC

#include <dolfin/log/dolfin_log.h>
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SLEPcEigenSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SLEPcEigenSolver::SLEPcEigenSolver()
{
  // Set up solver environment
  EPSCreate(PETSC_COMM_SELF, &eps);
}
//-----------------------------------------------------------------------------
SLEPcEigenSolver::~SLEPcEigenSolver()
{
  // Destroy solver environment
  if (eps) EPSDestroy(eps);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(const PETScMatrix& A)
{
  solve(&A, 0, A.size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(const PETScMatrix& A, uint n)
{
  solve(&A, 0, n);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(const PETScMatrix& A, const PETScMatrix& B)
{
  solve(&A, &B, A.size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(const PETScMatrix& A, const PETScMatrix& B, uint n)
{
  solve(&A, &B, n);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::getEigenvalue(real& lr, real& lc)
{
  getEigenvalue(lr, lc, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::getEigenpair(real& lr, real& lc, PETScVector& r, PETScVector& c)
{
  getEigenpair(lr, lc, r, c, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::getEigenvalue(real& lr, real& lc, uint i)
{
  const int ii = static_cast<int>(i);

  // Get number of computed values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
    EPSGetValue(eps, ii, &lr, &lc);
  else
    error("Requested eigenvalue has not been computed");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::getEigenpair(real& lr, real& lc,
                                    PETScVector& r, PETScVector& c,
                                    uint i)
{
  const int ii = static_cast<int>(i);

  // Get number of computed eigenvectors/values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if (ii < num_computed_eigenvalues)
    EPSGetEigenpair(eps, ii, &lr, &lc, r.vec(), c.vec());
  else
    error("Requested eigenvalue/vector has not been computed");
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::solve(const PETScMatrix* A,
                             const PETScMatrix* B,
                             uint n)
{
  // Associate matrix (matrices) with eigenvalue solver
  dolfin_assert(A);
  dolfin_assert(A->size(0) == A->size(1));
  if (B)
  {
    dolfin_assert(B->size(0) == B->size(1) && B->size(0) == A->size(0));
    EPSSetOperators(eps, A->mat(), B->mat());
  }
  else
  {
    EPSSetOperators(eps, A->mat(), PETSC_NULL);
  }

  // Set number of eigenpairs to compute
  dolfin_assert(n <= A->size(0));
  const uint nn = static_cast<int>(n);
  EPSSetDimensions(eps, nn, PETSC_DECIDE);

  // Set algorithm type (Hermitian matrix)
  //EPSSetProblemType(eps, EPS_NHEP);

  // Set options from database
  EPSSetFromOptions(eps);

  // Solve
  EPSSolve(eps);

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(eps, &reason);
  if (reason < 0)
  {
    warning("Eigenvalue solver did not converge");
    return;
  }

  // Report solver status
  int num_iterations = 0;
  EPSGetIterationNumber(eps, &num_iterations);
  EPSType eps_type = 0;
  EPSGetType(eps, &eps_type);
  message("Eigenvalue solver (%s) converged in %d iterations.",
          eps_type, num_iterations);
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::readParameters()
{
  if (has("eigenvalue spectrum"))
    setSpectrum(get("eigenvalue spectrum"));
  if (has("eigenvalue solver"))
    setSolver(get("eigenvalue solver"));
  if (has("eigenvalue tolerance") && has("eigenvalue iterations"))
    setTolerance(get("eigenvalue tolerance"), get("eigenvalue iterations"));
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::setSpectrum(std::string spectrum)
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
  else
  {
    warning("Requested spectrum unknown. Using largest magnitude.");
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
  }

  // FIXME: Need to add some test here as most algorithms only compute
  // FIXME: largest eigenvalues. Asking for smallest leads to a PETSc error.
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::setSolver(std::string solver)
{
  // Do nothing if default type is specified
  if (solver == "default")
    return;

  // Choose solver
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
  {
    warning("Requested Krylov method unknown. Using Krylov-Schur.");
    EPSSetType(eps, EPSKRYLOVSCHUR);
  }
}
//-----------------------------------------------------------------------------
void SLEPcEigenSolver::setTolerance(real tolerance, uint maxiter)
{
  dolfin_assert(tolerance > 0.0);
  EPSSetTolerances(eps, tolerance, static_cast<int>(maxiter));
}
//-----------------------------------------------------------------------------

#endif
