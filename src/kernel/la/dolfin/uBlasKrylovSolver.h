// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed:

#ifndef __UBLAS_KRYLOV_SOLVER_H
#define __UBLAS_KRYLOV_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/Parametrized.h>
#include <dolfin/DenseVector.h>
#include <dolfin/uBlasSparseMatrix.h>

namespace dolfin
{
  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b using uBlas data types. 

  // FIXME: Decide whether to implement a KrylovSolver for this class
  
  class uBlasKrylovSolver : public Parametrized
  {
  public:

    /// Krylov methods
    enum Type
    { 
      bicgstab,       // Stabilised biconjugate gradient squared method 
      cg,             // Conjugate gradient method
      default_solver, // Default PETSc solver (use when setting solver from command line)
      gmres           // GMRES method
    };

    /// Create Krylov solver with default method and preconditioner
    uBlasKrylovSolver();

    /// Create Krylov solver for a particular method with default preconditioner
    uBlasKrylovSolver(Type solver);

//    /// Create Krylov solver with default method and a particular preconditioner
//    uBlasKrylovSolver(Preconditioner::Type preconditioner);

//    /// Create Krylov solver with default method and a particular preconditioner
//    uBlasKrylovSolver(Preconditioner& preconditioner);

//    /// Create Krylov solver for a particular method and preconditioner
//    uBlasKrylovSolver(Type solver, Preconditioner::Type preconditioner);

//    /// Create Krylov solver for a particular method and preconditioner
//    uBlasKrylovSolver(Type solver, Preconditioner& preconditioner);

    /// Destructor
    ~uBlasKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    template < class MAT >
    uint solve(const MAT& A, DenseVector& x, const DenseVector& b)
      {
        dolfin_warning("Krylov solvers for uBlas data types have not been implemented.");
        dolfin_warning("LU solver will be used. This may be slow and consume a lot of memory.");
        A.solve(x, b);
        return 1;
      }
          
  private:

    /// Krylov solver type
    Type type;

  };

}

#endif
