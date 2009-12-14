// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-14
// Last changed: 2009-12-14

#ifndef __LAPACK_SOLVERS_H
#define __LAPACK_SOLVERS_H

namespace dolfin
{

  class LAPACKMatrix;
  class LAPACKVector;

  /// This class provides a simple interface to selected LAPACK
  /// solvers.

  class LAPACKSolvers
  {
  public:

    // Solve least squares system in-place (by calling LAPACK DGELSS)
    static void solve_least_squares(const LAPACKMatrix& A, LAPACKVector& b);

  };

}

#endif
