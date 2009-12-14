// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-14
// Last changed: 2009-12-14

#ifndef __LAPACK_VECTOR_H
#define __LAPACK_VECTOR_H

#include <dolfin/common/types.h>

namespace dolfin
{

  /// This class provides a simple wrapper for matrix data for use
  /// with LAPACK (column-major ordering).
  ///
  /// This class does currently not implement the GenericVector
  /// interface but may possibly be extended to do so in the future.

  class LAPACKVector
  {
  public:

    /// Create M x N matrix
    LAPACKVector(uint M) : M(M), values(new double[M]) {}

    /// Destructor
    ~LAPACKVector()
    { delete [] values; }

    /// Return size of vector
    uint size() const
    { return M; }

    /// Access entry i
    double& operator[] (uint i)
    { return values[i]; }

    /// Access entry i, const version
    double operator[] (uint i) const
    { return values[i]; }

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
