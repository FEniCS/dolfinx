// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005
// Last changed: 2008-04-22

#ifdef HAS_PETSC

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScKrylovMatrix.h"
#include "PETScLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver()
  : ksp(0), B(0), idxm(0), idxn(0)
{
  init();
}
//-----------------------------------------------------------------------------
PETScLUSolver::~PETScLUSolver()
{
  clear();
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const GenericMatrix& A, GenericVector& x, 
                                       const GenericVector& b) 
{
  return solve(A.down_cast<PETScMatrix>(), x.down_cast<PETScVector>(), 
               b.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScMatrix& A, PETScVector& x, 
                                  const PETScVector& b)
{
  // Initialise solver
  init();

  MatType mat_type;
  MatGetType(A.mat(), &mat_type);

  // Convert to UMFPACK matrix if matrix type is MATSEQAIJ and UMFPACK is available.
  #if PETSC_HAVE_UMFPACK
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
  x.resize(A.size(1));

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

  // Clear data
  clear();

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScKrylovMatrix& A,
		       PETScVector& x, const PETScVector& b)
{
  // Initialise solver
  init();

  // Get parameters
  const bool report = get("LU report");

  // Copy data to dense matrix
  const double Anorm = copyToDense(A);
  
  // Initialize solution vector (remains untouched if dimensions match)
  x.resize(A.size(1));

  // Write a message
  if ( report )
    message("Solving linear system of size %d x %d (PETSc LU solver).",
		A.size(0), A.size(1));

  // Solve linear system
  KSPSetOperators(ksp, B, B, DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, b.vec(), x.vec());

  // Estimate condition number for l1 norm
  const double xnorm = x.norm(l1);
  const double bnorm = b.norm(l1) + DOLFIN_EPS;
  const double kappa = Anorm * xnorm / bnorm;
  if ( kappa > 0.001 / DOLFIN_EPS )
  {
    if ( kappa > 1.0 / DOLFIN_EPS )
      error("Matrix has very large condition number (%.1e). Is it singular?", kappa);
    else
      warning("Matrix has large condition number (%.1e).", kappa);
  }

  // Clear data
  clear();

  return 1;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::disp() const
{
  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
double PETScLUSolver::copyToDense(const PETScKrylovMatrix& A)
{
  error("PETScLUSolver::copyToDense needs to be fixed");

  return 0;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::init()
{
  // Set up solver environment to use only preconditioner
  KSPCreate(PETSC_COMM_SELF, &ksp);
  
  // Set preconditioner to LU factorization
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCLU);

  // Allow matrices with zero diagonals to be solved
  PCFactorSetShiftNonzero(pc, PETSC_DECIDE);
}
//-----------------------------------------------------------------------------
#endif
