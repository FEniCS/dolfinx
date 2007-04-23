// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
// 
// First added:  2006-08-15
// Last changed: 2006-08-16

#ifndef __GMRES_H
#define __GMRES_H

#include <dolfin/PETScKrylovSolver.h>
#include <dolfin/uBlasKrylovSolver.h>

namespace dolfin
{

  /// This class provides methods for solving a linear system with
  /// the GMRES method, with an optional preconditioner.
  
  class GMRES
  {
  public:

#ifdef HAVE_PETSC_H
    
    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b,
		      Preconditioner pc = default_pc);

    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b,
		      Preconditioner pc = default_pc);

    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b,
		      PETScPreconditioner& pc);
         
    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b,
		      PETScPreconditioner& pc);
    
#endif

    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b,
		      Preconditioner pc = default_pc);
    
    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b,
		      Preconditioner pc = default_pc);

    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const uBlasKrylovMatrix& A, uBlasVector& x, const uBlasVector& b,
		      Preconditioner pc = default_pc);
    
    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b,
		      uBlasPreconditioner& pc);
    
    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b,
		      uBlasPreconditioner& pc);

    /// Solve linear system Ax = b and return number of iterations
    static uint solve(const uBlasKrylovMatrix& A, uBlasVector& x, const uBlasVector& b,
		      uBlasPreconditioner& pc);
    
  };

}

#endif
