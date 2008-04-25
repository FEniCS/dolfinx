// Copyright (C) 2004-2008 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006-2007.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2004-01-01
// Last changed: 2008-04-22

#ifndef __PETSC_MATRIX_H
#define __PETSC_MATRIX_H

#ifdef HAS_PETSC

#include <petscmat.h>

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include "PETScObject.h"
#include "GenericMatrix.h"

namespace dolfin
{

  class PETScVector;

  /// This class provides a simple matrix class based on PETSc.
  /// It is a simple wrapper for a PETSc matrix pointer (Mat)
  /// implementing the GenericMatrix interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class PETScMatrix : public GenericMatrix, public PETScObject, public Variable
  {
  public:

    /// PETSc sparse matrix types
    enum Type
    {
      default_matrix, // Default matrix type
      spooles,        // Spooles
      superlu,        // Super LU
      umfpack         // UMFPACK
    };

    /// Create empy matrix
    explicit PETScMatrix(Type type=default_matrix);

    /// Create M x N matrix
    explicit PETScMatrix(uint M, uint N, Type type=default_matrix);

    /// Copy constructor
    explicit PETScMatrix(const PETScMatrix& A);

    /// Create matrix from given PETSc Mat pointer
    explicit PETScMatrix(Mat A);

    /// Destructor
    ~PETScMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    PETScMatrix* copy() const;

    /// Return size of given dimension
    uint size(uint dim) const;

    /// Set all entries to zero and keep any sparse structure
    void zero();

    /// Finalize assembly of tensor
    void apply();

    /// Display tensor
    void disp(uint precision=2) const;

    //--- Implementation of the GenericMatrix interface --

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Get block of values
    void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    void set(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    void add(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Get non-zero values of given row
    void getrow(uint row, Array<uint>& columns, Array<real>& values) const;

    /// Set given rows to zero
    void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    void ident(uint m, const uint* rows);

    // Matrix-vector product, y = Ax
    void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const;

    /// Multiply matrix by given number
    const PETScMatrix& operator*= (real a);

    /// Assignment operator
    const GenericMatrix& operator= (const GenericMatrix& A);

    /// Assignment operator
    const PETScMatrix& operator= (const PETScMatrix& A);

    //--- Convenience functions ---

    /// Divide matrix by given number
    virtual const PETScMatrix& operator/= (real a);

    //--- Special functions ---

    /// Return linear algebra backend factory
    LinearAlgebraFactory& factory() const;

    //--- Special PETScFunctions ---

    /// Return PETSc Mat pointer
    Mat mat() const;

    /// Return PETSc matrix type 
    Type type() const;

    /// Return norm of matrix
    enum Norm {l1, linf, frobenius};
    real norm(const Norm type = l1) const;

  private:

    // Initialize M x N matrix with a given number of nonzeros per row
    void init(uint M, uint N, const uint* nz);

    // Initialize M x N matrix with a given number of nonzeros per row diagonal and off-diagonal
    void init(uint M, uint N, const uint* d_nzrow, const uint* o_nzrow);

    // Set PETSc matrix type
    void setType();

    // Return PETSc matrix type
    MatType getPETScType() const;

    // Check that requested type has been compiled into PETSc
    void checkType();

    // PETSc Mat pointer
    Mat A;

    // True if the pointer is a copy of someone else's data
    bool _copy;

    // PETSc matrix type
    Type _type;

  };

  /// Output of PETScMatrix
  LogStream& operator<< (LogStream& stream, const PETScMatrix& A);

}

#endif

#endif
