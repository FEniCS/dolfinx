// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEWGMRES_H
#define __NEWGMRES_H

#include <dolfin/constants.h>

namespace dolfin
{
  
  class NewMatrix;
  class NewVector;

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

  private:

    // Tolerance
    real tol;

    // Maximum number of iterations
    unsigned int maxiter;

  };

}

#endif
