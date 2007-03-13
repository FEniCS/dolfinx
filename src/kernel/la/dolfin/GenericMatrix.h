// Copyright (C) 2006-2007 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2006.
// Modified by Anders Logg 2006.
//
// First added:  2006-04-24
// Last changed: 2007-03-13

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <dolfin/constants.h>

namespace dolfin
{
  class SparsityPattern;

  /// This class defines a common interface for sparse and dense matrices.

  class GenericMatrix
  {
  public:
 
    /// Constructor
    GenericMatrix() {}

    /// Destructor
    virtual ~GenericMatrix() {}

    /// Initialize M x N matrix
    virtual void init(const uint M, const uint N) = 0;

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    virtual void init(const uint M, const uint N, const uint nzmax) = 0;
    
    /// Initialize M x N matrix with given number of nonzeros per row
    virtual void init(const uint M, const uint N, const int nz[]) = 0;

    /// Initialize a matrix from the sparsity pattern
    virtual void init(const SparsityPattern& sparsity_pattern) = 0;

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    virtual uint size(const uint dim) const = 0;

    /// Access element value
    virtual real get(const uint i, const uint j) const = 0;

    /// Set element value
    virtual void set(const uint i, const uint j, const real value) = 0;

    /// Set block of values
    virtual void set(const real block[], const int rows[], const int m, const int cols[], const int n) = 0;

    /// Add block of values
    virtual void add(const real block[], const int rows[], const  int m, const int cols[], const int n) = 0;

    /// Apply changes to matrix (only needed for sparse matrices)
    virtual void apply() = 0;

    /// Set all entries to zero
    virtual void zero() = 0;

    /// Set given rows to identity matrix
    virtual void ident(const int rows[], const int m) = 0;

    /// Return maximum number of nonzero entries in all rows
    virtual uint nzmax() const = 0;
    
  };

}

#endif
