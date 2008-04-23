// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
// Modified by Ola Skavhaug, 2007.
//
// First added:  2007-03-15
// Last changed: 2008-04-22

#ifndef __SCALAR_H
#define __SCALAR_H

#include <dolfin/common/types.h>
#include <dolfin/log/LogStream.h>
#include "GenericTensor.h"
#include "uBlasFactory.h"

namespace dolfin
{

  class GenericSparsityPattern;

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
    void init(uint rank, const uint* dims)
    { value = 0.0; }

    /// Return copy of tensor
    virtual Scalar* copy() const
    { Scalar* s = new Scalar(); s->value = value; return s; }

    /// Initialize zero tensor using sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern)
    { value = 0.0; }

    /// Return rank of tensor (number of dimensions)
    uint rank() const 
    { return 0; }

    /// Return size of given dimension
    uint size(uint dim) const
    { return 1; }

    /// Get block of values
    void get(real* block, const uint* num_rows, const uint * const * rows) const
    { block[0] = value; }

    /// Set block of values
    void set(const real* block, const uint* num_rows, const uint * const * rows)
    { value = block[0]; }
    
    /// Add block of values
    void add(const real* block, const uint* num_rows, const uint * const * rows)
    { value += block[0]; }

    /// Set all entries to zero and keep any sparse structure (implemented by sub class)
    void zero()
    { value = 0.0; }

    /// Finalise assembly of tensor
    void apply() {}

    /// Display tensor
    void disp(uint precision = 2) const
    { cout << "Scalar value: " << value << endl; }

    ///--- Scalar interface ---
    
    /// Cast to real
    operator real()
    { return value; }

    /// Assignment from real
    const Scalar& operator=(real value)
    { this->value = value; 
      return *this; 
    }

    /// Get value (needed for SWIG interface)
    real getval() const 
    { return value; }

    /// Return a factory for the default linear algebra backend
    inline LinearAlgebraFactory& factory() const 
    { return dolfin::uBlasFactory::instance(); }

  private:
    
    real value;

  };

}

#endif
