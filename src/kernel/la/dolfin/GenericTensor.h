// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

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

    /// Add entries to tensor
    virtual void add(real* block, uint* size, uint** entries) = 0;

  };

}

#endif
