// Copyright (C) 2006-2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2006-05-15
// Last changed: 2007-12-07

#ifndef __MATRIX_H
#define __MATRIX_H

#include "GenericMatrix.h"
#include <dolfin/main/dolfin_main.h>

#include "default_la_types.h"

namespace dolfin
{
  
  /// This class provides an interface to the default DOLFIN
  /// matrix implementation as decided in default_la_types.h.

  // FIXME: This class should exactly duplicate the GenericMatrix interface
  // FIXME: Nothing more or less (perhaps with exception from the mat() function
  
  class Matrix : public GenericMatrix, public Variable
  {
  public:
    
    /// Constructor
    Matrix() : GenericMatrix(), Variable("A", "DOLFIN matrix"),
	       matrix(new DefaultMatrix()) {}
    
    /// Constructor
    Matrix(uint M, uint N) : GenericMatrix(), Variable("A", "DOLFIN matrix"),
			     matrix(new DefaultMatrix(M, N)) {}
    
    /// Destructor
    ~Matrix() { delete matrix; }
    
    /// Initialize M x N matrix
    inline void init(uint M, uint N) 
    { matrix->init(M, N); }
    
    /// Initialize zero matrix using sparsity pattern
    inline void init(const GenericSparsityPattern& sparsity_pattern)
    { matrix->init(sparsity_pattern); }
    
    /// Create uninitialized matrix
    inline Matrix* create() const
    { return new Matrix(); }

    /// Create copy of matrix
    inline Matrix* copy() const
    {
      Matrix* Mcopy = create();
      Mcopy->matrix = matrix->copy();
      return Mcopy;
//       DefaultMatrix* mcopy = matrix->copy();
//       delete Mcopy->matrix;
//       Mcopy->matrix = mcopy;
    }

    /// Return size of given dimension
    inline uint size(uint dim) const
    { return matrix->size(dim); }

    /// Get block of values
    inline void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const
    { matrix->get(block, m, rows, n, cols); }
    
    /// Set block of values
    inline void set(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix->set(block, m, rows, n, cols); }
    
    /// Add block of values
    inline void add(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix->add(block, m, rows, n, cols); }

    /// Set all entries to zero and keep any sparse structure (implemented by sub class)
    inline void zero()
    { matrix->zero(); }
    
    /// Set given rows to identity matrix
    inline void ident(uint m, const uint* rows)
    { matrix->ident(m, rows); }
        
    /// Finalise assembly of matrix
    inline void apply()
    { matrix->apply(); }
    
    /// Display matrix (sparse output is default)
    void disp(uint precision = 2) const
    { matrix->disp(precision); }

    /// FIXME: Functions below are not in the GenericVector interface.
    /// FIXME: Should these be removed or added to the interface?

    /// Get non-zero values of row i
    inline void getRow(uint i, int& ncols, Array<int>& columns, Array<real>& values)
    { matrix->getRow(i, ncols, columns, values); }
    
    /// Return const reference to implementation
    inline const DefaultMatrix& mat() const
    { return *matrix; }
    
    /// Return const reference to implementation
    inline DefaultMatrix& mat()
    { return *matrix; }

    inline LinearAlgebraFactory& factory() const
    { return matrix->factory(); }
    
  private:

    // FIXME: Why should this be static? Why not just GenericMatrix*? (envelope-letter)
    // FIXME: And why a pointer here and not in Vector?
    DefaultMatrix* matrix;
    
  };

}

#endif
