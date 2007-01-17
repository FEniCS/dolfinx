// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

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

  };

}

#endif
