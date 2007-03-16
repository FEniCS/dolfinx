// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-01-17
// Last changed: 2007-03-16

#ifndef __GENERIC_TENSOR_H
#define __GENERIC_TENSOR_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// This class defines a common interface for general tensors.

  class GenericTensor
  {
  public:

    /// Constructor
    GenericTensor() {};

    /// Destructor
    virtual ~GenericTensor() {}

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, uint* dims) = 0;

    /// Return size of given dimension
    virtual uint size(const uint dim) const = 0;

    /// Add block of values
    virtual void add(real* block, uint* num_rows, uint** rows) = 0;

    /// Finalise assembly of tensor
    virtual void apply() = 0;

  };

}

#endif
