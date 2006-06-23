// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-06-23

#ifndef __UBLAS_KRYLOV_SOLVER_H
#define __UBLAS_KRYLOV_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/ublas.h>
#include <dolfin/LinearSolver.h>
#include <dolfin/Parametrized.h>
#include <dolfin/uBlasPreconditioner.h>


namespace dolfin
{
  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b using uBlas data types. 

  class DenseVector;
  class uBlasSparseMatrix;

  class uBlasKrylovSolver : public LinearSolver, public Parametrized
  {
  public:

    /// Krylov methods
    enum Type
    { 
      bicgstab,       // Stabilised biconjugate gradient squared method 
      cg,             // Conjugate gradient method (no yet implemented)
      default_solver, // Default PETSc solver (use when setting solver from command line)
      gmres           // GMRES method
    };

    /// Create Krylov solver with default method and preconditioner
    uBlasKrylovSolver();

    /// Create Krylov solver for a particular method with default preconditioner
    uBlasKrylovSolver(Type solver);

    /// Destructor
    ~uBlasKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const uBlasSparseMatrix& A, DenseVector& x, const DenseVector& b);
          
  private:

    /// Solve linear system Ax = b using restarted GMRES
    uint gmresSolver(const uBlasSparseMatrix& A, DenseVector& x, const DenseVector& b,
                     const uBlasPreconditioner& P, bool& converged) const;

    /// Solve linear system Ax = b using BiCGStab
    uint bicgstabSolver(const uBlasSparseMatrix& A, DenseVector& x, const DenseVector& b,
                        const uBlasPreconditioner& P, bool& converged) const;

    /// Read solver parameters
    void readParameters();

    /// Preconditioner
    uBlasPreconditioner P;

    /// Krylov solver type
    Type type;

    /// Solver parameters
    real rtol, atol, div_tol;
    uint max_it, restart;
    bool report;

    /// True if we have read parameters
    bool parameters_read;

  };

}

#endif
