// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson 2006.
// Modified by Anders Logg 2006-2007.
// Modified by Ola Skavhaug 2007.
// Modified by Kent-Andre Mardal 2008.
//
// First added:  2006-04-24
// Last changed: 2008-03-21

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <dolfin/main/constants.h>
#include "GenericTensor.h"

namespace dolfin
{

  class GenericVector; 
  class GenericSparsityPattern;
  template<class M>
  class Array;
  
  /// This class defines a common interface for matrices.
  
  class GenericMatrix : public GenericTensor
  {
  public:
    
    /// Constructor
    GenericMatrix() : GenericTensor() {}

    /// Destructor
    virtual ~GenericMatrix() {}
    
    ///--- Implementation of GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern (implemented by sub class)
    virtual void init(const GenericSparsityPattern& sparsity_pattern) = 0;

    // FIXME: Not needed?
    /// Create uninitialized matrix
    virtual GenericMatrix* create() const = 0;

    /// Create copy of matrix
    virtual GenericMatrix* copy() const = 0;

    /// Return rank of tensor (number of dimensions)
    inline uint rank() const { return 2; }

    /// Return size of given dimension (implemented by sub class)
    virtual uint size(uint dim) const = 0;

    /// Get block of values
    inline void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set block of values
    inline void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Add block of values
    inline void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set all entries to zero and keep any sparse structure (implemented by sub class)
    virtual void zero() = 0;

    /// Finalise assembly of tensor (implemented by sub class)
    virtual void apply() = 0;

    /// Display tensor (implemented by sub class)
    virtual void disp(uint precision = 2) const = 0;

    ///--- Matrix interface ---

    /// Initialize M x N matrix
    virtual void init(uint M, uint N) = 0;
    
    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const = 0;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows) = 0;

	// FIXME: Add more functions here

    /// Set given matrix rows to zero
    virtual void zero(uint m, const uint* rows) = 0;

    /// Set given matrix entry to value
    virtual void set(uint i, uint j, real value) {
      set(&value, 1, &i, 1, &j);  
    }

    // y = A x  ( or y = A^T x if transposed==true) 
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const = 0; 

    // FIXME remove this function and re-implement the << operator in terms of sparsity pattern and get
    /// Get non-zero values of row i
    virtual void getRow(uint i, int& ncols, Array<int>& columns, Array<real>& values) const = 0;



  };

}

#endif
