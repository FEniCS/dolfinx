// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003

#ifndef __MATRIX_H
#define __MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Variable.h>
#include <dolfin/General.h>

namespace dolfin {

  class Vector;
  class GenericMatrix;

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
    enum Type {sparse, dense, generic};

    /// Create an empty matrix
    Matrix(Type type = sparse);

    /// Create matrix with given dimensions
    Matrix(int m, int n, Type type = sparse);

    /// Create a copy of a given matrix
    Matrix(const Matrix& A);

    /// Destructor
    virtual ~Matrix();

    // Forward declaration of nested classes
    class Element;
    class Row;
    class Column;

    ///--- Basic operations

    /// Initialize to a zero matrix with given dimensions
    virtual void init(int m, int n);

    /// Clear all data
    virtual void clear();

    ///--- Simple functions
    
    /// Return matrix type, sparse or dense
    virtual Type type() const;

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    virtual int size(int dim) const;

    /// Return number of non-zero elements (only sparse)
    virtual int size() const;

    /// Return number of non-zero elements on row i (only sparse)
    virtual int rowsize(int i) const;

    /// Return size of matrix in bytes (approximately)
    virtual int bytes() const;

    ///--- Operators

    /// Index operator
    virtual real operator()(int i, int j) const;

    /// Index operator
    virtual Element operator()(int i, int j);

    /// Index operator
    virtual Row operator()(int i, Range j);

    /// Index operator
    virtual Row operator()(Index i, Range j);
    
    /// Index operator
    virtual Column operator()(Range i, int j);

    /// Index operator
    virtual Column operator()(Range i, Index j);

    /// Index operator (only dense, quick access)
    virtual real* operator[](int i) const;

    /// Index operator (only sparse, quick access)
    virtual real operator()(int i, int& j, int pos) const;

    /// Assignment from scalar (affects only already non-zero elements for sparse)
    virtual void operator=(real a);

    /// Assignment from a given matrix
    virtual void operator=(const Matrix& A);

    /// Add a given matrix
    virtual void operator+=(const Matrix& A);

    /// Subtract a given matrix
    virtual void operator-=(const Matrix& A);

    /// Multiplication with a given scalar
    virtual void operator*=(real a);    
    
    ///--- Matrix operations

    /// Compute maximum norm
    virtual real norm() const;

    /// Matrix-vector multiplication, component i of Ax
    virtual real mult(const Vector& x, int i) const;

    /// Matrix-vector multiplication
    virtual void mult(const Vector& x, Vector& Ax) const;
    
    /// Matrix-vector multiplication with transpose
    virtual void multt(const Vector& x, Vector &Ax) const;

    /// Scalar product with row
    virtual real multrow(const Vector& x, int i) const;

    /// Scalar product with column
    virtual real multcol(const Vector& x, int j) const;

    /// Compute transpose
    virtual void transp(Matrix& At) const;

    /// Solve Ax = b (in-place LU for dense and Krylov for sparse)
    virtual void solve(Vector& x, const Vector& b);

    /// Compute inverse (only dense)
    virtual void inverse(Matrix& Ainv);

    /// Solve Ax = b with high precision (only dense, not in-place)
    virtual void hpsolve(Vector& x, const Vector& b) const;

    /// Compute LU factorization (only dense, in-place)
    virtual void lu();

    /// Solve A x = b using a computed lu factorization (only dense)
    virtual void solveLU(Vector& x, const Vector& b) const;

    /// Compute inverse using a computed lu factorization (only dense)
    virtual void inverseLU(Matrix& Ainv) const;

    /// Solve A x = b with high precision using a computed lu factorization (only dense)
    virtual void hpsolveLU(const Matrix& LU, Vector& x, const Vector& b) const;
    
    /// --- Special functions

    /// Clear unused elements (only sparse)
    virtual void resize();
    
    /// Set A(i,j) = d_{ij} on row i
    virtual void ident(int i);

    /// Add a new row
    virtual void addrow();
    
    /// Add a new row given by a vector
    virtual void addrow(const Vector& x);

    /// Specify number of non-zero elements on row i (only sparse)
    virtual void initrow(int i, int rowsize);

    /// True if we have reached the end of the row (only sparse)
    virtual bool endrow(int i, int pos) const;
    
    /// Set this matrix to the transpose of the given matrix
    virtual void settransp(const Matrix& A);

    /// --- Output

    /// Display entire matrix
    virtual void show() const;

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
    
      Row(Matrix& matrix, int i, Range j);
      Row(Matrix& matrix, Index i, Range j);

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
      Range j;

    };

    /// Nested class Column, reference to a column in the matrix
    class Column {
    public:
    
      Column(Matrix& matrix, Range i, int j);
      Column(Matrix& matrix, Range i, Index j);

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
      Range i;
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
