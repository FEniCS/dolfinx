// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson, 2006.
// Modified by Anders Logg, 2006-2008.
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2006-04-24
// Last changed: 2008-08-07

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <boost/tuple/tuple.hpp>
#include "GenericTensor.h"

namespace dolfin
{

  class GenericVector;
  template<class M> class Array;

  /// This class defines a common interface for matrices.

  class GenericMatrix : public GenericTensor
  {
  public:

    /// Destructor
    virtual ~GenericMatrix() {}
    
    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor with given dimensions
    virtual void init(uint rank, const uint* dims)
    { dolfin_assert(rank == 2); init(dims[0], dims[1]); }

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern) = 0;

    /// Return copy of tensor
    virtual GenericMatrix* copy() const = 0;

    /// Return tensor rank (number of dimensions)
    virtual uint rank() const
    { return 2; }

    /// Return size of given dimension
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

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply() = 0;

    /// Display tensor
    virtual void disp(uint precision=2) const = 0;

    //--- Matrix interface ---

    /// Initialize M x N matrix
    virtual void init(uint M, uint N) = 0;

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const = 0;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols) = 0;

    /// Get non-zero values of given row
    virtual void getrow(uint row, Array<uint>& columns, Array<real>& values) const = 0;

    /// Set values for given row
    virtual void setrow(uint row, const Array<uint>& columns, const Array<real>& values) = 0;

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows) = 0;

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows) = 0;

    /// Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const = 0;

    /// Multiply matrix by given number
    virtual const GenericMatrix& operator*= (real a) = 0;

    /// Divide matrix by given number
    virtual const GenericMatrix& operator/= (real a) = 0;

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& x) = 0;

    /// Return pointers to underlying compresssed storage data
    /// FIXME: Document what data each of the three pointers points to.
    virtual boost::tuple<const std::size_t*, const std::size_t*, const double*, int> data() const
    { 
      error("Unable to return pointers to underlying matrix data."); 
      return boost::tuple<const std::size_t*, const std::size_t*, const double*, int>(0, 0, 0, 0);
    } 

    //--- Convenience functions ---

    /// Get value of given entry 
    virtual real operator() (uint i, uint j) const
    { real value(0); get(&value, 1, &i, 1, &j); return value; }

    /// Get value of given entry 
    virtual real getitem(std::pair<uint, uint> ij) const
    { real value(0); get(&value, 1, &ij.first, 1, &ij.second); return value; }

    /// Set given entry to value
    virtual void setitem(std::pair<uint, uint> ij, real value)
    { set(&value, 1, &ij.first, 1, &ij.second); }

  };

}
#endif
