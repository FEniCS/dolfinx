// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21

#ifndef __EPETRA_MATRIX_H
#define __EPETRA_MATRIX_H

#ifdef HAS_TRILINOS

#include <Epetra_CrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Variable.h>
#include "GenericMatrix.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  /// Forward declarations
  //class EpetraVector;
  class GenericSparsityPattern;

  /// This class represents a sparse matrix of dimension M x N.
  /// It is a simple wrapper for a Epetra matrix pointer (Epetra_FECrsMatrix).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Epetra_FECrsMatrix pointer using the function mat() and
  /// use the standard Epetra interface.

  class EpetraMatrix: public GenericMatrix, public Variable
  {
  public:
    
    /// Empty matrix
    EpetraMatrix();

    /// Create matrix of given size
    EpetraMatrix(uint M, uint N);

    /// Create matrix from given Epetra_FECrsMatrix pointer
    explicit EpetraMatrix(Epetra_FECrsMatrix* A);

    /// Create matrix from given Epetra_CrsGraph pointer
    explicit EpetraMatrix(const Epetra_CrsGraph& graph);

    /// Destructor
    virtual ~EpetraMatrix();

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Initialize a matrix from the sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern); 

    /// Create uninitialized matrix
    EpetraMatrix* create() const;

    /// Create copy of matrix
    EpetraMatrix* copy() const;

    /// Return number of rows (dim=0) or columns (dim=1) along dimension dim
    uint size(uint dim) const;
   
    /// Get block of values
    void get(real* block, 
	     uint m, const uint* rows, 
	     uint n, const uint* cols) const;

    /// Set block of values
    void set(const real* block, 
	     uint m, const uint* rows, 
	     uint n, const uint* cols);

    /// Add block of values
    void add(const real* block, 
	     uint m, const uint* rows, 
	     uint n, const uint* cols);

    /// Set all entries to zero
    void zero();

    /// Apply changes to matrix
    void apply();

    /// Display matrix (sparse output is default)
    void disp(uint precision = 2) const;

    /// Multiply matrix by given number
    const EpetraMatrix& operator*= (real a);

    /// Assignment operator
    const GenericMatrix& operator= (const GenericMatrix& x)
    { error("Not implemented."); return *this; }

    /// Assignment operator
    const EpetraMatrix& operator= (const EpetraMatrix& x)
    { error("Not implemented."); return *this; }


    /// Set given rows to identity matrix
    void ident(uint m, const uint* rows);

    /// Set given rows to zero matrix
    void zero(uint m, const uint* rows);

    // y = A x  ( or y = A^T x if transposed==true) 
    void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const; 

    /// Get non-zero values of row i
    void getrow(uint i, int& ncols, Array<int>& columns, Array<real>& values) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, 
				  const Epetra_FECrsMatrix& A);
    
    /// Return factory object for backend
    LinearAlgebraFactory& factory() const;

    /// Return Epetra_FECrsMatrix pointer
    Epetra_FECrsMatrix& mat() const;

  private:

    // Epetra_FECrsMatrix pointer
    Epetra_FECrsMatrix* A;
    
    // True if the pointer is a copy of someone else's data
    bool _copy;
    
  };

  LogStream& operator<< (LogStream& stream, const Epetra_FECrsMatrix& A);

}

#endif
#endif
