// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
// 
// First added:  2006-08-16
// Last changed: 2006-08-16

#ifndef __LU_H
#define __LU_H

#include "LUSolver.h"

namespace dolfin
{

  /// This class provides methods for solving a linear system by
  /// LU factorization.
  
  class LU
  {
  public:

    /// Solve linear system Ax = b
    static void solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);
    
  };

}

#endif
