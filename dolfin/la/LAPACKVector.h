// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-14
// Last changed: 2009-12-15

#ifndef __LAPACK_VECTOR_H
#define __LAPACK_VECTOR_H

#include <string>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  /// This class provides a simple wrapper for matrix data for use
  /// with LAPACK (column-major ordering).
  ///
  /// This class does currently not implement the GenericVector
  /// interface but may possibly be extended to do so in the future.

  class LAPACKVector : public Variable
  {
  public:

    /// Create M x N matrix
    LAPACKVector(uint M);

    /// Destructor
    ~LAPACKVector();

    /// Return size of vector
    uint size() const
    { return M; }

    /// Access entry i
    double& operator[] (uint i)
    { return values[i]; }

    /// Access entry i, const version
    double operator[] (uint i) const
    { return values[i]; }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

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
