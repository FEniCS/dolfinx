// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-08-31
// Last changed: 2006-09-21

#ifdef HAVE_SLEPC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/SLEPcEigenvalueSolver.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/PETScVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SLEPcEigenvalueSolver::SLEPcEigenvalueSolver(): eps(0), type(default_solver)
{
  // Set up solver environment
  EPSCreate(PETSC_COMM_SELF, &eps);

  // Set which eigenvalues to compute
  set("Eigenvalues to compute", "largest");

}
//-----------------------------------------------------------------------------
SLEPcEigenvalueSolver::SLEPcEigenvalueSolver(Type solver): eps(0), type(solver)
{
  // Set up solver environment
  EPSCreate(PETSC_COMM_SELF, &eps);

  // Set which eigenvalues to compute
  set("Eigenvalues to compute", "largest");

}
//-----------------------------------------------------------------------------
SLEPcEigenvalueSolver::~SLEPcEigenvalueSolver()
{
  // Destroy solver environment
  if ( eps ) 
    EPSDestroy(eps);
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::solve(const PETScMatrix& A)
{
  solve(A, 0, A.size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::solve(const PETScMatrix& A, uint n)
{
  solve(A, 0, n);
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::solve(const PETScMatrix& A, const PETScMatrix& B)
{
  solve(A, &B, A.size(0));
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::solve(const PETScMatrix& A, const PETScMatrix& B, uint n)
{
  solve(A, &B, n);
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::getEigenvalue(real& xr, real& xc)
{
  getEigenvalue(xr, xc, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::getEigenpair(real& xr, real& xc, PETScVector& r,  PETScVector& c)
{
  getEigenpair(xr, xc, r, c, 0);
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::getEigenvalue(real& xr, real& xc, const int i)
{
  // Get number of computed values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if( i < num_computed_eigenvalues )
    EPSGetValue(eps, i, &xr, &xc);
  else
    error("Requested eigenvalue has not been computed");
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::getEigenpair(real& xr, real& xc, PETScVector& r, PETScVector& c, const int i)
{
  // Get number of computed eigenvectors/values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if( i < num_computed_eigenvalues )
    EPSGetEigenpair(eps, i, &xr, &xc, r.vec(), c.vec());
  else
    error("Requested eigenvalue/vector has not been computed");
}
//-----------------------------------------------------------------------------
void SLEPcEigenvalueSolver::solve(const PETScMatrix& A, const PETScMatrix* B, uint n)
{
  const std::string eigenvalues_compute = get("Eigenvalues to compute");

  dolfin_assert( A.size(0) == A.size(1) );

  // Associate matrix (matrices) with eigenvalue solver
  if ( B )
  {
    dolfin_assert( B->size(0) == B->size(1) && B->size(0) == A.size(0) );
    EPSSetOperators(eps, A.mat(), B->mat());
  }
  else
    EPSSetOperators(eps, A.mat(), PETSC_NULL);
  
  // Set number of eigenpairs to compute
  dolfin_assert( n <= A.size(0));
  EPSSetDimensions(eps, n, PETSC_DECIDE);

  // Compute n largest eigenpairs
  if (eigenvalues_compute == "largest")
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);

// FIXME: Need to add some test here as most algorithms only compute largest eigenvalues
//        Asking for smallest leads to a PETSc error.
//  else if (eigenvalues_compute == "smallest")
//    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
//  else
//    error("Invalid choice if which eigenvalues to compute (smallest/largest)");
  
  // Set algorithm type
  EPSType eps_type = getType(type);
  if(eps_type != "default")
    EPSSetType(eps, eps_type);

//  // Set algorithm type (Hermitian matrix)
//  EPSSetProblemType(eps, EPS_HEP);

  // Set options
  EPSSetFromOptions(eps);

  // Solve
  EPSSolve(eps);  

  // Check for convergence
  EPSConvergedReason reason;
  EPSGetConvergedReason(eps, &reason);
  if( reason < 0 )
    warning("Eigenvalue solver did not converge"); 

  // Get number of iterations
  int num_iterations;
  EPSGetIterationNumber(eps, &num_iterations);

  // Get algorithm type
  EPSGetType(eps, &eps_type);

  message("Eigenvalue solver (%s) converged in %d iterations.",
	      eps_type, num_iterations);
}
//-----------------------------------------------------------------------------
EPSType SLEPcEigenvalueSolver::getType(const Type type) const
{
  switch (type)
  {
  case arnoldi:
    return EPSARNOLDI;
  case default_solver:
    return "default";
  case lanczos:
    return EPSLANCZOS;
  case lapack:
    return EPSLAPACK;
  case power:
    return EPSPOWER;
  case subspace:
    return EPSSUBSPACE;
  default:
    warning("Requested Krylov method unknown. Using GMRES.");
    return KSPGMRES;
  }
}
//-----------------------------------------------------------------------------

#endif
