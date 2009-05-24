// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrom, 2008.
// Modified by Anders Logg, 2009.
//
// First added:  2007-11-30
// Last changed: 2009-05-23

#ifndef __GENERIC_SPARSITY_PATTERN_H
#define __GENERIC_SPARSITY_PATTERN_H

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  /// Base class (interface) for generic tensor sparsity patterns.
  /// Currently, this interface is mostly limited to matrices.

  class GenericSparsityPattern : public Variable
  {
  public:

    /// Create empty sparsity pattern
    GenericSparsityPattern() {}

    /// Destructor
    virtual ~GenericSparsityPattern() {};

    /// Initialize sparsity pattern for a generic tensor
    virtual void init(uint rank, const uint* dims) = 0;

    /// Insert non-zero entries
    virtual void insert(const uint* num_rows, const uint * const * rows) = 0;

    /// Sort entries for each row 
    virtual void sort() = 0;

    /// Return rank
    virtual uint rank() const = 0;

    /// Return global size for dimension i
    virtual uint size(uint i) const = 0;

    /// Return local range
    virtual std::pair<uint, uint> range() const = 0;

    /// Return total number of nonzeros in local rows
    virtual uint num_nonzeros() const = 0;

    /// Fill array with number of nonzeros per local row for diagonal block
    virtual void num_nonzeros_diagonal(uint* num_nonzeros) const = 0;

    /// Fill array with number of nonzeros per local row for off-diagonal block
    virtual void num_nonzeros_off_diagonal(uint* num_nonzeros) const = 0;

    /// Finalize sparsity pattern
    virtual void apply() = 0;

  };

}

#endif
