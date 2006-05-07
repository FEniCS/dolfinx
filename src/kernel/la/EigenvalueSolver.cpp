// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
// 
// First added:  2005-08-31
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/EigenvalueSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EigenvalueSolver::EigenvalueSolver()
{
  // Initialize PETSc
  PETScManager::init();
  
  // Set up solver environment
  KSPCreate(PETSC_COMM_SELF, &ksp);
}
//-----------------------------------------------------------------------------
EigenvalueSolver::~EigenvalueSolver()
{
  // Destroy solver environment
  if ( ksp ) KSPDestroy(ksp);

}
//-----------------------------------------------------------------------------
void EigenvalueSolver::eigen(const Matrix& A, Vector& r,  Vector& c)
{
  // Dummy vectors needed for solve before eigenvalues can be computed
  Vector a, b;

  uint m = A.size(0);
  uint n = A.size(1);

  // Check that matrix A is square
  if(m != n)
  {
    dolfin_info("Matrix not square. Eigenvalues not being computed");
    return; 
	}

  // Initialize vectors for eigenvalues and dummy vectors
  r.init(n);
  c.init(n);
  a.init(n);
  b.init(n);

  // Solve linear system with trivial RHS (needed for eigenvalue solve)
  KSPSetOperators(ksp, A.mat(), A.mat(), SAME_NONZERO_PATTERN);
  KSPSolve(ksp, a.vec(), b.vec());

  // Allocate memory for eigenvalues
  real* er    = new real[n];
  real* ec    = new real[n];
  int*  block = new int[n];

  // Compute n eigenvalues using explicit algorithm
  dolfin_info("Computing all eigenvalues directly. Use only for small systems.");
  KSPComputeEigenvaluesExplicitly(ksp, n, er, ec);  

  // Add the computed real and complex eigenvalues into vectors 
  r = 0.0;
  c = 0.0;
  for(uint i=0; i< n; ++i) *(block+i) = i; 
  r.add(er, block, n);
  c.add(ec, block, n);
 
  delete [] er;
  delete [] ec;
  delete [] block;
}
//-----------------------------------------------------------------------------

#endif
