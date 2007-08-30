// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-03-15
// Last changed: 2007-08-28

#ifndef __SCALAR_H
#define __SCALAR_H

#include <dolfin/constants.h>
#include <dolfin/LogStream.h>
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
    inline void init(uint rank, const uint* dims)
    { value = 0.0; }

    /// Initialize zero tensor using sparsity pattern
    inline void init(const SparsityPattern& sparsity_pattern)
    { value = 0.0; }

    /// Create uninitialized scalar
    inline Scalar* create() const
    { return new Scalar(); }

    /// Create copy of scalar
    inline Scalar* copy() const
    { Scalar* s = new Scalar(); s->value = value; return s; }

    /// Return rank of tensor (number of dimensions)
    inline uint rank() const { return 0; }

    /// Return size of given dimension
    inline uint size(uint dim) const
    { return 1; }

    /// Get block of values
    inline void get(real* block, const uint* num_rows, const uint * const * rows) const
    { block[0] = value; }

    /// Set block of values
    inline void set(const real* block, const uint* num_rows, const uint * const * rows)
    { value = block[0]; }
    
    /// Add block of values
    inline void add(const real* block, const uint* num_rows, const uint * const * rows)
    { value += block[0]; }

    /// Set all entries to zero and keep any sparse structure (implemented by sub class)
    inline void zero()
    { value = 0.0; }

    /// Finalise assembly of tensor
    inline void apply() {}

    /// Display tensor
    inline void disp(uint precision = 2) const
    { cout << "Scalar value: " << value << endl; }

    ///--- Scalar interface ---
    
    /// Cast to real
    inline operator real()
    { return value; }

    /// Assignment from real
    inline const Scalar& operator=(real value)
    { this->value = value; return *this; }

    /// Get value (needed for SWIG interface)
    inline real getval() const { return value; }

  private:
    
    real value;

  };

}

#endif
