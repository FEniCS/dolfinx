// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
// Modified by Ola Skavhaug, 2007.
//
// First added:  2007-01-17
// Last changed: 2007-12-07

#ifndef __GENERIC_TENSOR_H
#define __GENERIC_TENSOR_H

#include <dolfin/main/constants.h>

namespace dolfin
{

  class GenericSparsityPattern;
  class LinearAlgebraFactory;

  /// This class defines a common interface for general tensors.

  class GenericTensor
  {
  public:

    /// Constructor
    GenericTensor() {};

    /// Destructor
    virtual ~GenericTensor() {}

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern) = 0;

    /// Create uninitialized tensor
    virtual GenericTensor* create() const = 0;

    /// Create copy of tensor
    virtual GenericTensor* copy() const = 0;

    /// Return rank of tensor (number of dimensions)
    virtual uint rank() const = 0;

    /// Return size of given dimension
    virtual uint size(uint dim) const = 0;

    /// Get block of values
    virtual void get(real* block, const uint* num_rows, const uint * const * rows) const = 0;

    /// Set block of values
    virtual void set(const real* block, const uint* num_rows, const uint * const * rows) = 0;

    /// Add block of values
    virtual void add(const real* block, const uint* num_rows, const uint * const * rows) = 0;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalise assembly of tensor
    virtual void apply() = 0;

    /// Display tensor
    virtual void disp(uint precision = 2) const = 0;

    /// Get LA backend factory
    virtual LinearAlgebraFactory& factory() const = 0; 

  };

}

#endif
