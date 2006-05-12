// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-08-31
// Last changed: 

#ifdef HAVE_SLEPC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/EigenvalueSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EigenvalueSolver::EigenvalueSolver(): eps(0), type(default_solver)
{
  // Initialize PETSc/SLEPc
  PETScManager::init();

  // Set up solver environment
  EPSCreate(PETSC_COMM_SELF, &eps);

  // Set which eigenvalues to compute
  set("Eigenvalues to compute", "largest");

}
//-----------------------------------------------------------------------------
EigenvalueSolver::EigenvalueSolver(Type solver): eps(0), type(solver)
{
  // Initialize PETSc/SLEPc
  PETScManager::init();

  // Set up solver environment
  EPSCreate(PETSC_COMM_SELF, &eps);

  // Set which eigenvalues to compute
  set("Eigenvalues to compute", "largest");

}
//-----------------------------------------------------------------------------
EigenvalueSolver::~EigenvalueSolver()
{
  // Destroy solver environment
  if ( eps ) 
    EPSDestroy(eps);
}
//-----------------------------------------------------------------------------
void EigenvalueSolver::solve(const Matrix& A)
{
  // Compute all eigenvalues/vectors  
  solve(A, A.size(0));
}
//-----------------------------------------------------------------------------
void EigenvalueSolver::solve(const Matrix& A, const uint n)
{
  const std::string eigenvalues_compute = get("Eigenvalues to compute");

  // Associate matrix with eigenvalue solver
  EPSSetOperators(eps, A.mat(), PETSC_NULL);

  // Set number of eigenpairs to compute
  EPSSetDimensions(eps, n, PETSC_DECIDE);

  // Compute n largest eigenpairs
  if (eigenvalues_compute == "largest")
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);

// FIXME: Need to add some test here as most algorithms only compute largest eigenvalues
//        Asking for smallest leads to a PETSc error.
//  else if (eigenvalues_compute == "smallest")
//    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
//  else
//    dolfin_error("Invalid choice if which eigenvalues to compute (smallest/largest)");
  
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
    dolfin_warning("Eigenvalue solver did not converge"); 

  // Get number of iterations
  int num_iterations;
  EPSGetIterationNumber(eps, &num_iterations);

  // Get algorithm type
  EPSGetType(eps, &eps_type);

  dolfin_info("Eigenvalue solver (%s) converged in %d iterations.",
	      eps_type, num_iterations);
}
//-----------------------------------------------------------------------------
void EigenvalueSolver::getEigenpair(real& xr, real& xc, Vector& r,  Vector& c)
{
  getEigenpair(xr, xc, r, c, 0);
}
//-----------------------------------------------------------------------------
void EigenvalueSolver::getEigenpair(real& xr, real& xc, Vector& r,  Vector& c, const int i)
{
  // Get number of computed eigenvectors/values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if( i < num_computed_eigenvalues )
    EPSGetEigenpair(eps, i, &xr, &xc, r.vec(), c.vec());
  else
    dolfin_error("Requested eigenvalue/vector has not been computed");
}
//-----------------------------------------------------------------------------
void EigenvalueSolver::getEigenvalue(real& xr, real& xc)
{
  getEigenvalue(xr, xc, 0);
}
//-----------------------------------------------------------------------------
void EigenvalueSolver::getEigenvalue(real& xr, real& xc, const int i)
{
  // Get number of computed values
  int num_computed_eigenvalues;
  EPSGetConverged(eps, &num_computed_eigenvalues);

  if( i < num_computed_eigenvalues )
    EPSGetValue(eps, i, &xr, &xc);
  else
    dolfin_error("Requested eigenvalue has not been computed");
}
//-----------------------------------------------------------------------------
EPSType EigenvalueSolver::getType(const Type type) const
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
    dolfin_warning("Requested Krylov method unknown. Using GMRES.");
    return KSPGMRES;
  }
}
//-----------------------------------------------------------------------------

#endif
