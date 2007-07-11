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

  class Matrix : public GenericMatrix, public Variable
  {
    /// This class defines an interface for a Matrix. This is the default
    /// matrix type is defined in default_la_types.h.
    
    public:

      Matrix() {}

      Matrix(uint i, uint j) : matrix(i,j) {}

      ~Matrix() {}

      inline void init(uint i, uint j) 
        { matrix.init(i,j); }

      inline void init(const SparsityPattern& sparsity_pattern)
        { matrix.init(sparsity_pattern); }

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

      inline const DefaultMatrix& mat() const
        { return matrix; }

      inline DefaultMatrix& mat()
        { return matrix; }

    private:

      DefaultMatrix matrix;
  };
}

#endif
