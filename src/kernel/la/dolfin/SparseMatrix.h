// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SPARSE_MATRIX_HH
#define __SPARSE_MATRIX_HH

#include <iostream>

#include <dolfin/dolfin_constants.h>

namespace dolfin {
  
  class DenseMatrix;
  class Vector;
  
  class SparseMatrix{
  public:
	 
	 SparseMatrix  ();
	 SparseMatrix  (int m, int n);
	 ~SparseMatrix ();

	 /// Resize matrix
	 void resize(int m, int n);
	 /// Clear matrix
	 void clear();
	 /// Returns size (0 for rows, 1 for columns)
	 int size(int dim);
	 /// Returns number of nonzero elements
	 int size();
	 /// Returns size of matrix in bytes (approximately)
	 int bytes();
	 /// Set number of nonzero entries in row
	 void setRowSize(int i, int rowsize);
	 /// Returns size of row i
	 int rowSize(int i);
	 
	 /// Indexing: fast alternative
	 real operator()(int i, int *j, int pos) const;
	 /// Indexing: slow alternative
	 real& operator()(int i, int j);
	 real operator()(int i, int j) const;
		
	 /// Returns maximum norm
	 real norm();
	 /// Set all elements 0 on this row except (i,j) = 1
	 void setRowIdentity(int i);
	 /// Returns element i of Ax
	 real mult(Vector &x, int i);
	 /// Multiplies x with A and puts the result in Ax
	 void mult(Vector &x, Vector &Ax);

	 /// Output
	 void show();
	 friend ostream& operator << (ostream& output, SparseMatrix& sparseMatrix);
	 
  private:
	 
	 int m,n;

	 int  *rowsizes;
	 int  **columns;
	 real **values;
	 
  };

}
  
#endif
