// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-14
// Last changed: 2009-12-14

#ifndef __LAPACK_MATRIX_H
#define __LAPACK_MATRIX_H

#include <dolfin/common/types.h>

namespace dolfin
{

  /// This class provides a simple wrapper for matrix data for use
  /// with LAPACK (column-major ordering).
  ///
  /// This class does currently not implement the GenericMatrix
  /// interface but may possibly be extended to do so in the future.

  class LAPACKMatrix
  {
  public:

    /// Create M x N matrix
    LAPACKMatrix(uint M, uint N) : M(M), N(N), values(new double[N*M]) {}

    /// Destructor
    ~LAPACKMatrix()
    { delete [] values; }

    /// Return size of given dimension
    uint size(uint dim) const
    { assert(dim < 2); return (dim == 0 ? M : N); }

    /// Access entry (i, j)
    double& operator() (uint i, uint j)
    { return values[j*N +i]; }

    /// Access entry (i, j), const version
    double operator() (uint i, uint j) const
    { return values[j*N +i]; }

  private:

    // Friends
    friend class LAPACKSolvers;

    // Number of rows and columns
    uint M, N;

    // Values, stored column-major
    double* values;

  };

}

#endif
