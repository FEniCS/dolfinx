// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
// Modified by Ola Skavhaug 2008.
//
// First added:  2006-04-25
// Last changed: 2008-04-08

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include <dolfin/common/types.h>
#include "GenericSparsityPattern.h"
#include "GenericTensor.h"
#include <dolfin/log/log.h>

namespace dolfin
{

  /// This class defines a common interface for matrices.

  class GenericVector : public GenericTensor
  {
  public:
 
    /// Constructor
    GenericVector() : GenericTensor() {}

    /// Destructor
    virtual ~GenericVector() {}

    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor of given rank and dimensions
    inline void init(uint rank, const uint* dims, bool reset)
    { init(dims[0]); }

    /// Initialize zero tensor using sparsity pattern
    inline void init(const GenericSparsityPattern& sparsity_pattern)
    { init(sparsity_pattern.size(0)); }

    /// Return rank of tensor (number of dimensions)
    inline uint rank() const { return 1; }

    /// Return size of given dimension
    inline uint size(uint dim) const
    { return size(); }

    /// Get block of values
    inline void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0]); }

    /// Set block of values
    inline void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0]); }

    /// Add block of values
    inline void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0]); }

    /// Set all entries to zero and keep any sparse structure (implemented by sub class)
    virtual void zero() = 0;

    /// Finalise assembly of tensor (implemented by sub class)
    virtual void apply() = 0;

    /// Display tensor (implemented by sub class)
    virtual void disp(uint precision = 2) const = 0;

    ///--- Vector interface ---
    
    /// Initialize vector of size N
    virtual void init(uint N) = 0;

    /// Return size
    virtual uint size() const = 0;

    /// Get values
    virtual void get(real* values) const = 0;

    /// Set values
    virtual void set(real* values) = 0;

    /// Add values
    virtual void add(real* values) = 0;

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows) const = 0;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows) = 0;

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows) = 0;

    // FIXME: --- Work in progress below here ---
    // FIXME: Add more functions
    // FIXME: Cleanup
    // FIXME: Add itemwize get, set, add
    // FIXME: Add copy constructor and assignment operator

    /// Inner product 
    virtual real inner(const GenericVector& vector) const = 0; 

    //  this += a*x   
    virtual void axpy(real a, const GenericVector& x) = 0; 

    /// Assignment of vector
    virtual const GenericVector& operator= (const GenericVector& x) { 
      // Can not be abstract since a GenericVector reference is returned
      // Therefore the current implementation: 
      error("The operator = can not be used on GenericVector"); 
      return *this; 
    }

    /// Return const GenericVector* (internal library use only!)
    virtual const GenericVector* instance() const 
    { return this; }

    /// Return GenericVector* (internal library use only!)
    virtual GenericVector* instance() 
    { return this; }

  };  

}

#endif
