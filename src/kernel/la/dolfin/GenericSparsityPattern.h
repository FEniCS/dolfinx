// Copyright (C) 2007 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-11-30

#ifndef __GENERIC_SPARSITY_PATTERN_H
#define __GENERIC_SPARSITY_PATTERN_H

#include <dolfin/constants.h>

namespace dolfin
{
  /// Base class for sparsity patterns of vectors/matrices. Concrete sub classes can 
  /// be used to initalise vectors and sparse matrices.

  class GenericSparsityPattern
  {
  public:

    /// Constructor
    GenericSparsityPattern() {}

    /// Destructor
    virtual ~GenericSparsityPattern() {};
      
    /// Initialise sparsity pattern for a vector
    virtual void init(uint M) = 0;

    /// Initialise sparsity pattern for a matrix with total number of rows and columns
    virtual void init(uint M, uint N) = 0;

    /// Insert non-zero entry
    virtual void insert(uint row, uint column) = 0;

    /// Insert non-zero entry
    virtual void insert(uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Return global size 
    virtual uint size(uint n) const = 0;

    /// Return array with number of non-zeroes per row
    virtual void numNonZeroPerRow(uint nzrow[]) const = 0;

    /// Return total number of non-zeroes
    virtual uint numNonZero() const = 0;

  };

}

#endif
