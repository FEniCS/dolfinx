// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SPARSE_MATRIX_HH
#define __SPARSE_MATRIX_HH

#include "kw_constants.h"

class DenseMatrix;
class Vector;

/// Sparse data is stored in the following way:
/// 
///   locations: contains indices for where new rows start in <columns>
///   columns:   contains column numbers for non-zero entries
///   values:    contains the values stored row by row

class SparseMatrix{
public:

  SparseMatrix  ();
  SparseMatrix  (int m, int n, int *ncols);
  ~SparseMatrix ();

  /// Resize and reset the matrix
  void Resize(int m, int n, int *ncols);
  /// Reset matrix: elements zero, and columns -1
  void Reset();
  
  /// Set non-zero element number pos on row i, column j to val
  void Set(int i, int  j, int pos, real val);
  /// Set element (i,j) to val (requires some searching)
  void Set(int i, int  j, real val);
  /// Add val to element (i,j) (requires some searching)
  void Add(int i, int  j, real val);
  /// Get element pos on row i and its column number
  real Get(int i, int *j, int pos);
  /// Get the diagonal element on row i (0.0 if there is none)
  real GetDiagonal(int i);
  /// Copy the values to a dense matrix (only affects non-zero entries)
  void CopyTo(DenseMatrix *A);
    
  /// Compute norm of A
  real Norm();
  /// Get number of nonzeros in row i
  int GetRowLength(int i);
  /// Multiplies this matrix with B from the left, which gives AB 
  void Mult(SparseMatrix* B, SparseMatrix* AB);
  /// Multiplies matrix with x gives Ax 
  void Mult(Vector* x, Vector* Ax);
  /// Multiplies matrix with x gives element i of Ax 
  real Mult(int i, Vector* x);
  //// Get a copy of the matrix
  SparseMatrix *GetCopy();
  /// Transpose matrix
  void Transpose();
  /// Drop all elements below a given tolerance
  void DropZeros(real tol);
  /// Set row to 0 except diagonal to 1
  void SetRowIdentity(int i);
  /// Set rows to 0 except diagonal to 1
  void SetRowIdentity(int *rows, int no_rows);
  /// Multiply all elements in row i with a
  void ScaleRow(int i, real a);
  /// Display the matrix
  void Display();
  /// Display the matrix including all values
  void DisplayAll();
  /// Size(0) gives no rows, Size(1) gives no columns
  int Size(int i);

private:

  int m,n,size;
  
  int  *locations;
  int  *columns;
  real *values;

};

#endif
