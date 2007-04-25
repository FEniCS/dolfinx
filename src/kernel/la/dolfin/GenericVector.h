// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2007.
//
// First added:  2006-04-25
// Last changed: 2007-04-03

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include <dolfin/constants.h>
#include <dolfin/SparsityPattern.h>
#include <dolfin/GenericTensor.h>

namespace dolfin
{

  /// This class defines a common interface for sparse and dense vectors.

  class GenericVector : public GenericTensor
  {
  public:
 
    /// Constructor
    GenericVector() : GenericTensor() {}

    /// Destructor
    virtual ~GenericVector() {}

    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, const uint* dims, bool reset)
    { init(dims[0]); }

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const SparsityPattern& sparsity_pattern, bool reset)
    { init(sparsity_pattern.size(0)); }

    /// Return size of given dimension
    virtual uint size(uint dim) const
    { return size(); }

    /// Get block of values
    virtual void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0]); }

    /// Set block of values
    virtual void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0]); }

    /// Add block of values
    virtual void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0]); }

    /// Finalise assembly of tensor (implemented by sub class)
    virtual void apply() = 0;

    /// Display tensor (implemented by sub class)
    virtual void disp(uint precision = 2) const = 0;

    ///--- Vector interface ---
    
    /// Initialize vector of size N
    virtual void init(const uint N) = 0;

    /// Return size
    virtual uint size() const = 0;

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows) const = 0;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows) = 0;

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows) = 0;
    
    ///--- FIXME: Which of the functions below do we really need? ---

    /// Set all entries to zero
    virtual void zero() = 0;
    
    /// Compute sum of vector
    virtual real sum() const = 0;

  };  

}

#endif
