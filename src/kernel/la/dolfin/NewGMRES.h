// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __NEW_GMRES_H
#define __NEW_GMRES_H

#include <dolfin/constants.h>

namespace dolfin
{
  
  class NewMatrix;
  class NewVector;
  class VirtualMatrix;

  /// This is just a template. Write documentation here.
  
  class NewGMRES
  {
  public:

    /// Create GMRES solver
    NewGMRES();

    /// Destructor
    ~NewGMRES();

    /// Solve linear system Ax = b for a given right-hand side b
    void solve(const NewMatrix& A, NewVector& x, const NewVector& b);

    /// Solve linear system Ax = b for a given right-hand side b
    void solve(const VirtualMatrix& A, NewVector& x, const NewVector& b);

  private:

    // Tolerance
    real tol;

    // Maximum number of iterations
    unsigned int maxiter;

  };

}

#endif
