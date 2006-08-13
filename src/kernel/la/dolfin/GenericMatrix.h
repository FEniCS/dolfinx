// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-24
// Last changed: 2006-05-30

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// This class defines a common interface for sparse and dense matrices.

  class GenericMatrix
  {
  public:
 
    /// Constructor
    GenericMatrix() {}

    /// Destructor
    virtual ~GenericMatrix() {}

    /// Initialize M x N matrix
    virtual void init(uint M, uint N) = 0;

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    virtual void init(uint M, uint N, uint nzmax) = 0;
    
    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    virtual uint size(uint dim) const = 0;

    /// Set block of values
    virtual void set(const real block[], const int rows[], int m, const int cols[], int n) = 0;

    /// Add block of values
    virtual void add(const real block[], const int rows[], int m, const int cols[], int n) = 0;

    /// Apply changes to matrix (only needed for sparse matrices)
    virtual void apply() = 0;

    /// Set all entries to zero
    virtual void zero() = 0;

    /// Set given rows to identity matrix
    virtual void ident(const int rows[], int m) = 0;

    /// Return maximum number of nonzero entries in all rows
    virtual uint nzmax() const = 0;

    /// Element access
    virtual real get(int i, int j) = 0;
    virtual void set(int i, int j, real value) = 0;
    
  };

}

#endif
