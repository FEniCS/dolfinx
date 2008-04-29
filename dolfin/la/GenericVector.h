// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-04-25
// Last changed: 2008-04-29

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include "VectorNormType.h"
#include "GenericSparsityPattern.h"
#include "GenericTensor.h"

namespace dolfin
{

  /// This class defines a common interface for vectors.

  class GenericVector : public GenericTensor
  {
  public:

    /// Destructor
    virtual ~GenericVector() {}

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    inline void init(const GenericSparsityPattern& sparsity_pattern)
    { init(sparsity_pattern.size(0)); }

    /// Return copy of tensor
    virtual GenericVector* copy() const = 0;

    /// Return tensor rank (number of dimensions)
    inline uint rank() const
    { return 1; }

    /// Return size of given dimension
    inline uint size(uint dim) const
    { dolfin_assert(dim == 0); return size(); }

    /// Get block of values
    inline void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0]); }

    /// Set block of values
    inline void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0]); }

    /// Add block of values
    inline void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0]); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply() = 0;

    /// Display tensor
    virtual void disp(uint precision=2) const = 0;

    //--- Vector interface ---

    /// Initialize vector of size N
    virtual void init(uint N) = 0;

    /// Return size of vector
    virtual uint size() const = 0;

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows) const = 0;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows) = 0;

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows) = 0;

    /// Get all values
    virtual void get(real* values) const = 0;

    /// Set all values
    virtual void set(real* values) = 0;

    /// Add values to each entry
    virtual void add(real* values) = 0;

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(real a, const GenericVector& x) = 0;

    /// Return inner product with given vector
    virtual real inner(const GenericVector& x) const = 0;

    /// Return norm of vector
    virtual real norm(VectorNormType type=l2) const = 0;

    /// Return minimum value of vector
    virtual real min() const = 0;

    /// Return maximum value of vector
    virtual real max() const = 0;

    /// Multiply vector by given number
    virtual const GenericVector& operator*= (real a) = 0;

    /// Divide vector by given number
    virtual const GenericVector& operator/= (real a) = 0;

    /// Add given vector
    virtual const GenericVector& operator+= (const GenericVector& x) = 0;

    /// Subtract given vector
    virtual const GenericVector& operator-= (const GenericVector& x) = 0;

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x) = 0;

    /// Assignment operator
    virtual const GenericVector& operator= (real a) = 0;

    //--- Convenience functions ---

    /// Get value of given entry
    virtual real operator[] (uint i) const
    { real value(0); get(&value, 1, &i); return value; }

    /// Get value of given entry 
    virtual real getitem(uint i) const
    { real value(0); get(&value, 1, &i); return value; }

    /// Set given entry to value
    virtual void setitem(uint i, real value)
    { set(&value, 1, &i); }

  };

}

#endif
