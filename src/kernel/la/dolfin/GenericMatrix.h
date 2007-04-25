// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Anders Logg 2006-2007.
//
// First added:  2006-04-24
// Last changed: 2007-04-16

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <dolfin/constants.h>
#include <dolfin/GenericTensor.h>

namespace dolfin
{

  class SparsityPattern;
  
  /// This class defines a common interface for sparse and dense matrices.

  class GenericMatrix : public GenericTensor
  {
  public:
 
    /// Constructor
    GenericMatrix() : GenericTensor() {}

    /// Destructor
    virtual ~GenericMatrix() {}

    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, const uint* dims, bool reset = true)
    { init(dims[0], dims[1], reset); }

    /// Initialize zero tensor using sparsity pattern (implemented by sub class)
    virtual void init(const SparsityPattern& sparsity_pattern, bool reset = true) = 0;

    /// Return size of given dimension (implemented by sub class)
    virtual uint size(uint dim) const = 0;

    /// Get block of values
    virtual void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set block of values
    virtual void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Add block of values
    virtual void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Finalise assembly of tensor (implemented by sub class)
    virtual void apply() = 0;

    /// Display tensor (implemented by sub class)
    virtual void disp(uint precision = 2) const = 0;

    ///--- Matrix interface ---

    /// Initialize M x N matrix
    virtual void init(uint M, uint N, bool reset = true) = 0;
    
    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const = 0;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Set given rows to identity matrix
    virtual void ident(const uint rows[], uint m) = 0;

    ///--- FIXME: Which of the functions below do we really need? ---

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    virtual void init(uint M, uint N, uint nzmax) = 0;
    
    /// Initialize M x N matrix with given number of nonzeros per row
    virtual void init(uint M, uint N, const uint nz[]) = 0;

    /// Set all entries to zero
    virtual void zero() = 0;

  };

}

#endif
