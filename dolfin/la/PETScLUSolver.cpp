// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009-2010.
// Modified by Niclas Jansson, 2009.
//
// First added:  2005
// Last changed: 2010-02-15

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
namespace dolfin
{
  class PETScKSPDeleter
  {
  public:
    void operator() (KSP* ksp)
    {
      if (ksp)
        KSPDestroy(*ksp);
      delete ksp;
    }
  };
}
//-----------------------------------------------------------------------------
Parameters PETScLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("petsc_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver() : ksp(static_cast<KSP*>(0), PETScKSPDeleter()),
                                 B(0), idxm(0), idxn(0)
{
  // Set parameter values
  parameters = default_parameters();

  // Initialize PETSc LU solver
  init();
}
//-----------------------------------------------------------------------------
PETScLUSolver::~PETScLUSolver()
{
  if (B)
  {
    MatDestroy(B);
    B = 0;
  }
  delete [] idxm; idxm=0;
  delete [] idxn; idxn=0;
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

  const MatSolverPackage solver_type;
  PC pc;
  KSPGetPC(*ksp, &pc);
  PCFactorGetMatSolverPackage(pc, &solver_type);

  // Get parameters
  const bool report = parameters["report"];

  // Initialize solution vector (remains untouched if dimensions match)
  x.resize(A.size(1));

  // Write a message
  if (report)
    info("Solving linear system of size %d x %d (PETSc LU solver, %s).",
         A.size(0), A.size(1), solver_type);

  // Solve linear system
  KSPSetOperators(*ksp, *A.mat(), *A.mat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(*ksp, *b.vec(), *x.vec());

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScKrylovMatrix& A, PETScVector& x,
                                  const PETScVector& b)
{
  // Initialise solver
  init();

  // Get parameters
  const bool report = parameters["report"];

  // Copy data to dense matrix
  const double Anorm = copy_to_dense(A);

  // Initialize solution vector (remains untouched if dimensions match)
  x.resize(A.size(1));

  // Write a message
  if ( report )
    info("Solving linear system of size %d x %d (PETSc LU solver).",
		A.size(0), A.size(1));

  // Solve linear system
  KSPSetOperators(*ksp, B, B, DIFFERENT_NONZERO_PATTERN);
  KSPSolve(*ksp, *b.vec(), *x.vec());

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

  return 1;
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScLUSolver not implemented, calling PETSc KSPView directly.");
    KSPView(*ksp, PETSC_VIEWER_STDOUT_WORLD);
  }
  else
    s << "<PETScLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
double PETScLUSolver::copy_to_dense(const PETScKrylovMatrix& A)
{
  error("PETScLUSolver::copy_to_dense needs to be fixed");
  return 0.0;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::init()
{
  // We create a PETSc Krylov solver and instruct it to use LU preconditioner

  // Destroy old solver environment if necessary
  if (!ksp.unique())
    error("Cannot create new KSP Krylov solver. More than one object points to the underlying PETSc object.");

  ksp.reset(new KSP, PETScKSPDeleter());

  // Set up solver environment
  if (MPI::num_processes() > 1)
  {
    info("Creating parallel PETSc Krylov solver (for LU factorization).");
    KSPCreate(PETSC_COMM_WORLD, ksp.get());
  }
  else
    KSPCreate(PETSC_COMM_SELF, ksp.get());

  // Set preconditioner to LU factorization
  PC pc;
  KSPGetPC(*ksp, &pc);
  PCSetType(pc, PCLU);

  if (MPI::num_processes() == 1)
    PCFactorSetMatSolverPackage(pc, MAT_SOLVER_UMFPACK);
  else
  {
#if PETSC_HAVE_MUMPS
    PCFactorSetMatSolverPackage(pc, MAT_SOLVER_MUMPS);
#elif PETSC_HAVE_SPOOLES
    PCFactorSetMatSolverPackage(pc, MAT_SOLVER_SPOOLES);
#else
    error("No suitable solver for parallel LU. Consider configuring PETSc with MUMPS or SPOOLES.");
#endif
  }

  // Allow matrices with zero diagonals to be solved
  PCFactorSetShiftNonzero(pc, PETSC_DECIDE);

  // Do LU factorization in-place (saves memory)
  PCASMSetUseInPlace(pc);
}
//-----------------------------------------------------------------------------

#endif
