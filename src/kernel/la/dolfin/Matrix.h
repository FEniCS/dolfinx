// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003

#ifndef __MATRIX_H
#define __MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Variable.h>

namespace dolfin {

  class Vector;
  class GenericMatrix;

  /// A MatrixIndex is a special index for a Matrix. This is declared outside
  enum MatrixRange { all };
  enum MatrixIndex { first, last };

  /// Matrix with given number of rows and columns that can
  /// be either sparse (default) or dense.
  ///
  /// Some of the operations only work on one type of matrix,
  /// like the LU factorization which is implemented only for
  /// a dense matrix. Such functions are marked by
  ///
  ///     (only dense)
  /// or  (only sparse)
  ///
  /// Using a sparse function on a dense matrix (or the opposite)
  /// will give a warning or an error.

  class Matrix : public Variable {
  public:

    /// Matrix type, sparse or dense
    enum Type {DENSE, SPARSE};

    /// Create an empty matrix
    Matrix(Type type = SPARSE);

    /// Create matrix with given dimensions
    Matrix(int m, int n, Type type = SPARSE);

    /// Create a copy of a given matrix
    Matrix(const Matrix& A);

    /// Destructor
    ~Matrix ();

    // Forward declaration of nested classes
    class Element;
    class Row;
    class Column;

    ///--- Basic operations

    /// Initialize to a zero matrix with given dimensions
    void init(int m, int n);

    /// Clear all data
    void clear();

    ///--- Simple functions
    
    /// Return matrix type, sparse or dense
    Type type() const;

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    int size(int dim) const;

    /// Return number of non-zero elements (only sparse)
    int size() const;

    /// Return number of non-zero elements on row i (only sparse)
    int rowsize(int i) const;

    /// Return size of matrix in bytes (approximately)
    int bytes() const;

    ///--- Operators

    /// Index operator
    real operator()(int i, int j) const;

    /// Index operator
    Element operator()(int i, int j);

    /// Index operator
    Row operator()(int i, MatrixRange j);

    /// Index operator
    Row operator()(MatrixIndex i, MatrixRange j);
    
    /// Index operator
    Column operator()(MatrixRange i, int j);

    /// Index operator
    Column operator()(MatrixRange i, MatrixIndex j);

    /// Index operator (only dense, quick access)
    real* operator[](int i) const;

    /// Index operator (only sparse, quick access)
    real operator()(int i, int& j, int pos) const;

    /// Assignment from scalar (affects only already non-zero elements for sparse)
    void operator=(real a);

    /// Assignment from a given matrix
    void operator=(const Matrix& A);

    /// Add a given matrix
    void operator+=(const Matrix& A);

    /// Subtract a given matrix
    void operator-=(const Matrix& A);

    /// Multiplication with a given scalar
    void operator*=(real a);    
    
    ///--- Matrix operations

    /// Compute maximum norm
    real norm() const;

    /// Matrix-vector multiplication, component i of Ax
    real mult(const Vector& x, int i) const;

    /// Matrix-vector multiplication
    void mult(const Vector& x, Vector& Ax) const;
    
    /// Matrix-vector multiplication with transpose
    void multt(const Vector& x, Vector &Ax) const;

    /// Scalar product with row
    real multrow(const Vector& x, int i) const;

    /// Scalar product with column
    real multcol(const Vector& x, int j) const;

    /// Compute transpose
    void transp(Matrix& At) const;

    /// Solve Ax = b (in-place LU for dense and Krylov for sparse)
    void solve(Vector& x, const Vector& b);

    /// Compute inverse (only dense)
    void inverse(Matrix& Ainv);

    /// Solve Ax = b with high precision (only dense, not in-place)
    void hpsolve(Vector& x, const Vector& b) const;

    /// Compute LU factorization (only dense, in-place)
    void lu();

    /// Solve A x = b using a computed lu factorization (only dense)
    void solveLU(Vector& x, const Vector& b) const;

    /// Compute inverse using a computed lu factorization (only dense)
    void inverseLU(Matrix& Ainv) const;

    /// Solve A x = b with high precision using a computed lu factorization (only dense)
    void hpsolveLU(const Matrix& LU, Vector& x, const Vector& b) const;
    
    /// --- Special functions

    /// Clear unused elements (only sparse)
    void resize();
    
    /// Set A(i,j) = d_{ij} on row i
    void ident(int i);

    /// Add a new row
    void addrow();
    
    /// Add a new row given by a vector
    void addrow(const Vector& x);

    /// Specify number of non-zero elements on row i (only sparse)
    void initrow(int i, int rowsize);

    /// True if we have reached the end of the row (only sparse)
    bool endrow(int i, int pos) const;
    
    /// Set this matrix to the transpose of the given matrix
    void settransp(const Matrix& A);

    /// --- Output

    /// Display entire matrix
    void show() const;

    /// Condensed information in one line
    friend LogStream& operator<< (LogStream& stream, const Matrix& A);

    ///--- Nested classes

    /// Nested class Element, reference to a position in the matrix
    class Element {
    public:
    
      Element(Matrix& matrix, int i, int j);
      
      operator real() const;
      
      void operator=  (real a);
      void operator=  (const Element& e);
      void operator+= (real a);
      void operator-= (real a);
      void operator*= (real a);
      void operator/= (real a);
      

    protected:
    
      Matrix& A;
      int i;
      int j;

    };

    /// Nested class Row, reference to a row in the matrix
    class Row {
    public:
    
      Row(Matrix& matrix, int i, MatrixRange j);
      Row(Matrix& matrix, MatrixIndex i, MatrixRange j);

      int size() const;

      real operator() (int j) const;
      Element operator()(int j);

      void operator= (const Row& row);
      void operator= (const Column& col);
      void operator= (const Vector& x);

      real operator* (const Vector& x) const;
      
      friend class Column;

    private:
    
      Matrix& A;
      int i;
      MatrixRange j;

    };

    /// Nested class Column, reference to a column in the matrix
    class Column {
    public:
    
      Column(Matrix& matrix, MatrixRange i, int j);
      Column(Matrix& matrix, MatrixRange i, MatrixIndex j);

      int size() const;

      real operator() (int i) const;
      Element operator()(int i);

      void operator= (const Column& col);
      void operator= (const Row& row);
      void operator= (const Vector& x);

      real operator* (const Vector& x) const;

      friend class Row;

    private:
    
      Matrix& A;
      MatrixRange i;
      int j;

    };

    friend class DirectSolver;
    friend class Element;

  private:
    
    /// Return matrix values
    real** values();

    /// Return matrix values
    real** const values() const;
    
    /// Initialize permutation (only dense)
    void initperm();

    /// Clear permutation (only dense)
    void clearperm();

    /// Return permutation (only dense)
    int* permutation();

    /// Return permutation (only dense)
    int* const permutation() const;

    GenericMatrix* A;
    Type _type;

  };
  
}

#endif
