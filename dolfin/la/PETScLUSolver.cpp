// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2005
// Last changed: 2009-07-08

#ifdef HAS_PETSC

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/main/MPI.h>
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScKrylovMatrix.h"
#include "PETScLUSolver.h"
#include "LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters PETScLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("petsc_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver() : ksp(0), B(0), idxm(0), idxn(0)
{
  // Set parameter values
  parameters = default_parameters();

  // Initialize PETSc LU solver
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

  #if PETSC_VERSION_MAJOR > 2
  const MatSolverPackage solver_type;
  PC pc;
  KSPGetPC(ksp, &pc);
  PCFactorGetMatSolverPackage(pc, &solver_type);
  #else
  MatType solver_type;
  MatGetType(*A.mat(), &solver_type);
  #endif

  // Convert to UMFPACK matrix if matrix type is MATSEQAIJ and UMFPACK is available.
  #if PETSC_HAVE_UMFPACK && PETSC_VERSION_MAJOR < 3
  std::string _mat_type = solver_type;
  if (_mat_type == MATSEQAIJ)
  {
    Mat Atemp = *A.mat();
    MatConvert(*A.mat(), MATUMFPACK, MAT_REUSE_MATRIX, &Atemp);
  }
  #endif

  // Get parameters
  const bool report = parameters("report");

  // Initialize solution vector (remains untouched if dimensions match)
  x.resize(A.size(1));

  // Write a message
  if (report)
    info("Solving linear system of size %d x %d (PETSc LU solver, %s).",
         A.size(0), A.size(1), solver_type);

  // Solve linear system
  KSPSetOperators(ksp, *A.mat(), *A.mat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, *b.vec(), *x.vec());

  // Clear data
  clear();

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScKrylovMatrix& A, PETScVector& x,
                                  const PETScVector& b)
{
  // Initialise solver
  init();

  // Get parameters
  const bool report = parameters("report");

  // Copy data to dense matrix
  const double Anorm = copy_to_dense(A);

  // Initialize solution vector (remains untouched if dimensions match)
  x.resize(A.size(1));

  // Write a message
  if ( report )
    info("Solving linear system of size %d x %d (PETSc LU solver).",
		A.size(0), A.size(1));

  // Solve linear system
  KSPSetOperators(ksp, B, B, DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp, *b.vec(), *x.vec());

  // Estimate condition number for l1 norm
  const double xnorm = x.norm("l1");
  const double bnorm = b.norm("l1") + DOLFIN_EPS;
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
double PETScLUSolver::copy_to_dense(const PETScKrylovMatrix& A)
{
  error("PETScLUSolver::copy_to_dense needs to be fixed");

  return 0;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::init()
{
  // We create a PETSc Krylov solver and instruct it to use LU preconditioner

  // Set up solver environment
  if (MPI::num_processes() > 1)
  {
    info("Creating parallel PETSc Krylov solver (for LU factorization).");
    KSPCreate(PETSC_COMM_WORLD, &ksp);
  }
  else
  {
    KSPCreate(PETSC_COMM_SELF, &ksp);
  }

  // Set preconditioner to LU factorization
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCLU);

  #if PETSC_HAVE_UMFPACK && PETSC_VERSION_MAJOR > 2
  PCFactorSetMatSolverPackage(pc, MAT_SOLVER_UMFPACK);
  #endif

  // Allow matrices with zero diagonals to be solved
  PCFactorSetShiftNonzero(pc, PETSC_DECIDE);

  // Do LU factorization in-place (saves memory)
  PCASMSetUseInPlace(pc);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::clear()
{
  if (ksp)
  {
    KSPDestroy(ksp);
    ksp = 0;
  }

  if (B )
  {
    MatDestroy(B);
    ksp = 0;
  }

  delete [] idxm; idxm=0;
  delete [] idxn; idxn=0;
}
//-----------------------------------------------------------------------------

#endif
