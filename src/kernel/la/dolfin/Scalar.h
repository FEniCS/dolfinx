// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-15
// Last changed: 2007-03-15

#ifndef __SCALAR_H
#define __SCALAR_H

#include <dolfin/constants.h>
#include <dolfin/GenericTensor.h>

namespace dolfin
{

  /// This class represents a real-valued scalar quantity and
  /// implements the GenericTensor interface for scalars.
  
  class Scalar
  {
  public:

    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, uint* dims)
    { value = 0.0; }
    
    /// Return size of given dimension
    virtual uint size(const uint dim) const
    { return 1; }

    /// Add block of values
    virtual void add(real* block, uint* num_rows, uint** rows)
    { value += block[0]; }

    ///--- Scalar functions ---
    
    /// Cast to real
    inline operator real()
    { return value; }

    /// Assignment from real
    const Scalar& operator=(real value)
    { this->value = value; return *this; }

  private:
    
    real value;

  };

}

#endif
