// Copyright (C) 2006-2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-15
// Last changed: 2007-07-11

#ifndef __MATRIX_H
#define __MATRIX_H

#include <dolfin/GenericMatrix.h>
#include <dolfin/dolfin_main.h>

#include <dolfin/default_la_types.h>

namespace dolfin
{
  
  /// This class defines the interface for a standard Matrix using
  /// the default backend as decided in default_la_types.h.
  
  class Matrix : public GenericMatrix, public Variable
  {
  public:
    
    /// Constructor
    Matrix() : GenericMatrix(), Variable("A", "DOLFIN matrix") {}
    
    /// Constructor
    Matrix(uint M, uint N) : GenericMatrix(), Variable("A", "DOLFIN matrix"), matrix(M, N) {}
    
    /// Destructor
    ~Matrix() {}
    
    /// Initialize M x N matrix
    inline void init(uint M, uint N) 
    { matrix.init(M, N); }
    
    /// Initialize zero matrix using sparsity pattern
    inline void init(const SparsityPattern& sparsity_pattern)
    { matrix.init(sparsity_pattern); }
    
    /// Return size of given dimension
    inline uint size(const uint dim) const
    { return matrix.size(dim); }

    /// Get block of values
    inline void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const
    { matrix.get(block, m, rows, n, cols); }
    
    /// Set block of values
    inline void set(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix.set(block, m, rows, n, cols); }
    
    /// Add block of values
    inline void add(const real* block, const uint m, const uint* rows, const uint n, const uint* cols)
    { matrix.add(block, m, rows, n, cols); }
    
    /// Get non-zero values of row i
    inline void getRow(const uint i, int& ncols, Array<int>& columns, Array<real>& values)
    { matrix.getRow(i, ncols, columns, values); }
    
    /// Set given rows to identity matrix
    inline void ident(const uint rows[], uint m)
    { matrix.ident(rows, m); }
    
    /// Set all entries to zero
    inline void zero()
    { matrix.zero(); }
    
    /// Apply changes to matrix
    inline void apply()
    { matrix.apply(); }
    
    /// Display matrix (sparse output is default)
    void disp(uint precision = 2) const
    { matrix.disp(precision); }

    /// Return const reference to implementation
    inline const DefaultMatrix& mat() const
    { return matrix; }
    
    /// Return const reference to implementation
    inline DefaultMatrix& mat()
    { return matrix; }
    
    private:

      DefaultMatrix matrix;

  };

}

#endif
