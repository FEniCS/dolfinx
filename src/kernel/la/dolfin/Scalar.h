// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-15
// Last changed: 2007-04-03

#ifndef __SCALAR_H
#define __SCALAR_H

#include <dolfin/constants.h>
#include <dolfin/GenericTensor.h>

namespace dolfin
{

  class SparsityPattern;

  /// This class represents a real-valued scalar quantity and
  /// implements the GenericTensor interface for scalars.
  
  class Scalar : public GenericTensor
  {
  public:

    /// Constructor
    Scalar() : GenericTensor(), value(0.0) {}

    /// Destructor
    virtual ~Scalar() {}

    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor of given rank and dimensions
    inline void init(uint rank, uint* dims)
    { value = 0.0; }

    /// Initialize zero tensor using sparsity pattern
    inline void init(const SparsityPattern& sparsity_pattern)
    { value = 0.0; }
    
    /// Return size of given dimension
    inline uint size(const uint dim) const
    { return 1; }

    /// Get block of values
    inline void get(real* block, const uint* num_rows, const uint * const * rows) const
    { value = block[0]; }

    /// Set block of values
    inline void set(const real* block, const uint* num_rows, const uint * const * rows)
    { block[0] = value; }
    
    /// Add block of values
    inline void add(const real* block, const uint* num_rows, const uint * const * rows)
    { value += block[0]; }

    /// Finalise assembly of tensor
    inline void apply() {}

    ///--- Scalar interface ---
    
    /// Cast to real
    inline operator real()
    { return value; }

    /// Assignment from real
    inline const Scalar& operator=(real value)
    { this->value = value; return *this; }

  private:
    
    real value;

  };

}

#endif
