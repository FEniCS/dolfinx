// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_H
#define __ODE_H

#include <dolfin/Sparsity.h>
#include <dolfin/Vector.h>

namespace dolfin {

  /// ODE represents an initial value problem of the form
  ///
  ///     u'(t) = f(u(t),t)
  ///         
  ///     u(0)  = u0
  ///
  /// where u(t) is a vector of length N.

  class ODE {
  public:
    
    /// Constructor
    ODE(int N);

    /// Right-hand side
    virtual real f(const Vector& u, real t, int i) = 0;

    /// Number of components N
    int size() const;

    /// Solve ODE
    void solve();

    /// Set sparsity (number of dependencies for component i)
    void sparse(int i, int size);
    
    /// Set sparsity (component i depends on j)
    void depends(int i, int j);

    /// Set sparsity defined by a sparse matrix
    void sparse(const Matrix& A);
    
    /// Try to automatically detect dependencies
    void sparse();

    /// Sparsity
    Sparsity sparsity;
    
  protected:
    
    int N;
    real T;

    Vector u0;

  };

}

#endif
