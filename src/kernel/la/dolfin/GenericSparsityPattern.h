// Copyright (C) 2007 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrom 2008.
//
// First added:  2007-11-30
// Last changed: 2008-01-24

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

    /// Initialise sparsity pattern for a generic tensor
    virtual void init(uint rank, const uint* dims) = 0;

    /// Initialise sparsity pattern for a parallel generic tensor
    virtual void pinit(uint rank, const uint* dims) = 0;

    /// Insert non-zero entry
    virtual void insert(const uint* num_rows, const uint * const * rows) = 0;

    /// Insert non-zero entry
    virtual void pinsert(const uint* num_rows, const uint * const * rows) = 0;

    /// Return global size 
    virtual uint size(uint n) const = 0;

    /// Return array with number of non-zeroes per row
    virtual void numNonZeroPerRow(uint nzrow[]) const = 0;

    /// Return total number of non-zeroes
    virtual uint numNonZero() const = 0;

    /// Finalize sparsity pattern (needed by most parallel la backends)
    virtual void apply() = 0;

  };

}

#endif
