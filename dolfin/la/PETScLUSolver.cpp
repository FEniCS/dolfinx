// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005
// Last changed: 2006-08-24

#ifdef HAS_PETSC

#include <dolfin/log/dolfin_log.h>
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScKrylovMatrix.h"
#include "PETScLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver()
  : PETScLinearSolver(), ksp(0), B(0), idxm(0), idxn(0)
{
  // Set up solver environment to use only preconditioner
  KSPCreate(PETSC_COMM_SELF, &ksp);
  //KSPSetType(ksp, KSPPREONLY);
  
  // Set preconditioner to LU factorization
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCLU);

  // Allow matrices with zero diagonals to be solved
  PCFactorSetShiftNonzero(pc, PETSC_DECIDE);

  // Do LU factorization in-place (saves memory)
  PCASMSetUseInPlace(pc);
}
//-----------------------------------------------------------------------------
PETScLUSolver::~PETScLUSolver()
{
  if ( ksp ) KSPDestroy(ksp);
  if ( B ) MatDestroy(B);
  if ( idxm ) delete [] idxm;
  if ( idxn ) delete [] idxn;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScMatrix& A,
		       PETScVector& x, const PETScVector& b)
{
  MatType mat_type;

  // Convert to UMFPACK matrix if matrix type is MATSEQAIJ and UMFPACK is available.
  #if PETSC_HAVE_UMFPACK
    MatGetType(A.mat(), &mat_type);
    std::string _mat_type = mat_type;
    if(_mat_type == MATSEQAIJ)
    {
      Mat Atemp = A.mat();
      MatConvert(A.mat(), MATUMFPACK, MAT_REUSE_MATRIX, &Atemp);
    }
  #endif

  // Get parameters
  const bool report = get("LU report");

  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Write a message
  if ( report )
    message("Solving linear system of size %d x %d (PETSc LU solver, %s).",
            A.size(0), A.size(1), mat_type);

  // Solve linear system
  KSPSetOperators(ksp, A.mat(), A.mat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());
  
  // Get name of solver
  KSPType ksp_type;
  KSPGetType(ksp, &ksp_type);

  // Get name of preconditioner
  PC pc;
  KSPGetPC(ksp, &pc);
  PCType pc_type;
  PCGetType(pc, &pc_type);
  MatGetType(A.mat(), &mat_type);

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScKrylovMatrix& A,
		       PETScVector& x, const PETScVector& b)
{
  // Get parameters
  const bool report = get("LU report");

  //cout << "LU got matrix:" << endl;
  //cout << "A = "; A.disp(false);
  //cout << "b = "; b.disp();

  // Copy data to dense matrix
  const real Anorm = copyToDense(A);
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.init(A.size(1));

  // Write a message
  if ( report )
    message("Solving linear system of size %d x %d (PETSc LU solver).",
		A.size(0), A.size(1));

  // Solve linear system
  KSPSetOperators(ksp, B, B, DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Estimate condition number for l1 norm
  const real xnorm = x.norm(l1);
  const real bnorm = b.norm(l1) + DOLFIN_EPS;
  const real kappa = Anorm * xnorm / bnorm;
  if ( kappa > 0.001 / DOLFIN_EPS )
  {
    if ( kappa > 1.0 / DOLFIN_EPS )
      error("Matrix has very large condition number (%.1e). Is it singular?", kappa);
    else
      warning("Matrix has large condition number (%.1e).", kappa);
  }

  return 1;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::disp() const
{
  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
real PETScLUSolver::copyToDense(const PETScKrylovMatrix& A)
{
  error("PETScLUSolver::copyToDense needs to be fixed");
/*
  // Get size
  uint M = A.size(0);
  dolfin_assert(M = A.size(1));

  if ( !B )
  {
    // Create matrix if it has not been created before
    MatCreateSeqDense(PETSC_COMM_SELF, M, M, PETSC_NULL, &B);
    idxm = new int[M];
    idxn = new int[1];
    for (uint i = 0; i < M; i++)
      idxm[i] = i;
    idxn[0] = 0;
    e.init(M);
    y.init(M);

  }
  else
  {
    // Check dimensions of existing matrix
    int MM = 0;
    int NN = 0;
    MatGetSize(B, &MM, &NN);

    if ( MM != static_cast<int>(M) || NN != static_cast<int>(M) )
      error("FIXME: Matrix dimensions changed, not implemented (should be).");
  }

  // Multiply matrix with unit vectors to get the values
  real maxcolsum = 0.0;
  e = 0.0;
  for (uint j = 0; j < M; j++)
  {
    // Multiply with unit vector and set column
    e(j) = 1.0;
    A.mult(e, y);
    real* values = y.array(); // assumes uniprocessor case
    idxn[0] = j;
    MatSetValues(B, M, idxm, 1, idxn, values, INSERT_VALUES);
    y.restore(values);
    e(j) = 0.0;
    
    // Compute l1 norm of matrix (maximum column sum)
    const real colsum = y.norm(PETScVector::l1);
    if ( colsum > maxcolsum )
      maxcolsum = colsum;
  }

  MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

  //cout << "Copied to dense matrix:" << endl;
  //A.disp();
  //MatView(B, PETSC_VIEWER_STDOUT_SELF);

  return maxcolsum;
*/
  return 0;
}
//-----------------------------------------------------------------------------

#endif
