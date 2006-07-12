// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-05-31
// Last changed: 2006-07-03

#ifndef __UBLAS_KRYLOV_SOLVER_H
#define __UBLAS_KRYLOV_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/ublas.h>
#include <dolfin/Parametrized.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{

  /// Forward declarations
  class DenseVector;
  class uBlasKrylovMatrix;
  class uBlasPreconditioner;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b using uBlas data types.

  class uBlasKrylovSolver : public Parametrized, public LinearSolver
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

    /// Create Krylov solver with default method
    uBlasKrylovSolver();

    /// Create Krylov solver for a particular method
    uBlasKrylovSolver(Type solver);

    /// Destructor
    ~uBlasKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const uBlasKrylovMatrix& A, DenseVector& x, const DenseVector& b);

    /// Solve linear system Ax = b with preconditioning and return number of iterations
    uint solve(const uBlasKrylovMatrix& A, DenseVector& x, const DenseVector& b,
	       const uBlasPreconditioner& pc);
          
  private:

    /// Solve linear system Ax = b using restarted GMRES
    uint solveGMRES(const uBlasKrylovMatrix& A, DenseVector& x, const DenseVector& b,
		    const uBlasPreconditioner& pc, bool& converged) const;

    /// Solve linear system Ax = b using BiCGStab
    uint solveBiCGStab(const uBlasKrylovMatrix& A, DenseVector& x, const DenseVector& b,
		       const uBlasPreconditioner& pc, bool& converged) const;
    
    /// Read solver parameters
    void readParameters();

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
