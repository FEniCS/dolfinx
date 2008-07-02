// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-05-19

#ifndef __ASSEMBLY_MATRIX_H
#define __ASSEMBLY_MATRIX_H

#include <map>

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include "GenericMatrix.h"

namespace dolfin
{

  /// Simple implementation of a GenericMatrix for experimenting
  /// with new assembly. Not sure this will be used later but it
  /// might be useful.

  class SparsityPattern;

  class AssemblyMatrix : public GenericMatrix
  {
  public:

    /// Create empty matrix
    AssemblyMatrix();

    /// Create M x N matrix
    AssemblyMatrix(uint M, uint N);
  
    /// Copy constructor
    explicit AssemblyMatrix(const AssemblyMatrix& A);

    /// Destructor
    virtual ~AssemblyMatrix();

    ///--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    virtual AssemblyMatrix* copy() const;

    /// Return size of given dimension
    virtual uint size(uint dim) const;
    
    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(FinalizeType finaltype=FINALIZE);

    /// Display tensor
    virtual void disp(uint precision=2) const;

    //--- Implementation of the GenericMatrix interface ---
    
    /// Initialize M x N matrix
    virtual void init(uint M, uint N);

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /*
    /// Get block of values
    virtual void get(real* block, const uint* num_rows, const uint * const * rows) const
    { get(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set block of values
    virtual void set(const real* block, const uint* num_rows, const uint * const * rows)
    { set(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Add block of values
    virtual void add(const real* block, const uint* num_rows, const uint * const * rows)
    { add(block, num_rows[0], rows[0], num_rows[1], rows[1]); }
    */

    /// Get non-zero values of given row
    virtual void getrow(uint row, Array<uint>& columns, Array<real>& values) const;

    /// Set values for given row
    virtual void setrow(uint row, const Array<uint>& columns, const Array<real>& values);

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows);

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const;

    /// Multiply matrix by given number
    virtual const AssemblyMatrix& operator*= (real a);

    /// Divide matrix by given number
    virtual const AssemblyMatrix& operator/= (real a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A)
    { error("Not implemented."); return *this; }

    ///--- Specialized matrix functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    /// Initialize zero tensor of given rank and dimensions
    virtual void init(uint rank, const uint* dims, bool reset=true);

  private:

    // The matrix representation
    std::vector<std::map<uint, real> > A;

    // The size of the matrix
    uint* dims;

  };

}

#endif
