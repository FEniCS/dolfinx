// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-24
// Last changed: 

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <dolfin/constants.h>

namespace dolfin
{
  /// This template provdies a uniform interface to both dense and sparse 
  /// matrices.

  template < class T >
  class GenericMatrix 
  {
  public:
 
    /// Constructor
    GenericMatrix(){}

    /// Destructor
    ~GenericMatrix(){}

    /// Return the "leaf" object
    T& matrix() 
      { return static_cast<T&>(*this); }

    /// Return refernce to an entry
    real& operator() (uint i, uint j)
      { return matrix()(i,j); }

    /// Initialize M x N matrix
    void init(uint M, uint N)
      { matrix().init(M, N); }

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    void init(uint M, uint N, uint nzmax)
      { matrix().init(M, N, nzmax); }

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    uint size(uint dim)
      { return matrix().size(dim); }

    /// Set all entries to zero (want to replace this with GenericMatrix::clear())
    GenericMatrix& operator= (real zero)
      { return matrix() = zero; }

    /// Set all entries to zero (needs to be implemented for sparse matrices)
    GenericMatrix& clear()
      { return matrix().clear(); }

    /// Return maximum number of nonzero entries
    uint nzmax()
      { return matrix().nzmax(); }

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n)
      { matrix().add(block, rows, m, cols, n); }

    /// Apply changes to matrix (only needed for sparse matrices)
    void apply()
      { matrix().apply(); }

  private:

    
  };  
}
#endif
