// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

namespace dolfin
{

  class Matrix;
  class Vector;

  /// The class LinearSolver serves as a base class for all linear
  /// solvers and provides a number of utilites to simplify the
  /// implementation of iterative methods.
  ///
  /// A linear solver solves (approximately) a linear system Ax = b
  /// for a given right-hand side b.
  
  class LinearSolver
  {
  protected:
    
    /// Check data for linear system
    void check(const Matrix& A, Vector& x, const Vector& b) const;

    /// Compute l2-norm of residual
    real residual(const Matrix& A, Vector& x, const Vector& b) const;
    /// Compute l2-norm of residual and the residual vector
    real residual(const Matrix& A, Vector& x, const Vector& b, 
		  Vector& r) const;

    /// Iterative solution of the linear system. The virtual function
    /// iteration() is called in each iteration and should be implemented
    /// by a subclass making use of this function.
    void iterate(const Matrix& A, Vector& x, const Vector& b,
		 real tol, unsigned int maxiter);

    /// Perform one iteration on the linear system
    virtual void iteration(const Matrix& A, Vector& x, const Vector& b);

  };

}

#endif
