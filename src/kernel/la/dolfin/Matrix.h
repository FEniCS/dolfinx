// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003, 2004.

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
    Matrix(unsigned int m, unsigned int n, Type type = sparse);

    /// Create sparse matrix with given dimensions and number of nonzeros per row
    Matrix(unsigned int m, unsigned int n, unsigned int nz);

    /// Create a copy of a given matrix
    Matrix(const Matrix& A);

    /// Destructor
    virtual ~Matrix();

    // Forward declaration of nested classes
    class Element;
    class Row;
    class Column;

    ///--- Basic operations ---

    /// Initialize to a zero matrix with given dimensions
    virtual void init(unsigned int m, unsigned int n, Type type = sparse);

    /// Clear all data
    virtual void clear();

    ///--- Simple functions ---
    
    /// Return matrix type, sparse or dense
    virtual Type type() const;

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    virtual unsigned int size(unsigned int dim) const;

    /// Return number of non-zero elements (only sparse)
    virtual unsigned int size() const;

    /// Return number of non-zero elements on row i (only sparse)
    virtual unsigned int rowsize(unsigned int i) const;

    /// Return size of matrix in bytes (approximately)
    virtual unsigned int bytes() const;

    ///--- Operators ---

    /// Index operator
    virtual real operator()(unsigned int i, unsigned int j) const;

    /// Index operator
    virtual Element operator()(unsigned int i, unsigned int j);

    /// Index operator
    virtual Row operator()(unsigned int i, Range j);

    /// Index operator
    virtual Row operator()(Index i, Range j);
    
    /// Index operator
    virtual Column operator()(Range i, unsigned int j);

    /// Index operator
    virtual Column operator()(Range i, Index j);

    /// Index operator (only dense, quick access)
    virtual real* operator[](unsigned int i) const;

    /// Index operator (only sparse, quick access)
    virtual real operator()(unsigned int i, unsigned int& j, unsigned int pos) const;

    /// Index operator (only sparse, quick access)
    virtual real& operator()(unsigned int i, unsigned int& j, unsigned int pos);

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
    
    ///--- Matrix operations ---

    /// Compute maximum norm
    virtual real norm() const;

    /// Matrix-vector multiplication, component i of Ax
    virtual real mult(const Vector& x, unsigned int i) const;

    /// Matrix-vector multiplication
    virtual void mult(const Vector& x, Vector& Ax) const;
    
    /// Matrix-vector multiplication with transpose
    virtual void multt(const Vector& x, Vector &Ax) const;

    /// Matrix-matrix multiplication
    virtual void mult(const Matrix& B, Matrix& AB) const;

    /// Scalar product with row
    virtual real multrow(const Vector& x, unsigned int i) const;

    /// Scalar product with column
    virtual real multcol(const Vector& x, unsigned int j) const;

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

    /// Solve A x = b high precision using computed lu factorization (only dense)
    virtual void hpsolveLU(const Matrix& LU, Vector& x, const Vector& b) const;
    
    /// --- Special functions ---

    /// Clear unused elements (only sparse)
    virtual void resize();
    
    /// Set A(i,j) = d_{ij} on row i
    virtual void ident(unsigned int i);

    /// Compute lumped matrix and put result in the given vector
    virtual void lump(Vector& a) const;

    /// Add a new row
    virtual void addrow();
    
    /// Add a new row given by a vector
    virtual void addrow(const Vector& x);

    /// Specify number of non-zero elements on row i (only sparse)
    virtual void initrow(unsigned int i, unsigned int rowsize);

    /// True if we have reached the end of the row (only sparse)
    virtual bool endrow(unsigned int i, unsigned int pos) const;
    
    /// Set this matrix to the transpose of the given matrix
    virtual void settransp(const Matrix& A);

    /// Compute maximum element in given row
    virtual real rowmax(unsigned int i) const;

    /// Compute maximum element in given row
    virtual real colmax(unsigned int i) const;    
    
    /// Compute minimum element in given row
    virtual real rowmin(unsigned int i) const;

    /// Compute minimum element in given column
    virtual real colmin(unsigned int i) const;

    /// Compute sum of given row
    virtual real rowsum(unsigned int i) const;

    /// Compute sum of given row
    virtual real colsum(unsigned int i) const;

    /// Compute norm of given row: 0 for inf, 1 for l1, and 2 for l2
    virtual real rownorm(unsigned int i,unsigned int type) const;

    /// Compute norm of given column: 0 for inf, 1 for l1, and 2 for l2
    virtual real colnorm(unsigned int i,unsigned int type) const;

    ///--- Output ---

    /// Display entire matrix
    virtual void show() const;

    /// Condensed information in one line
    friend LogStream& operator<< (LogStream& stream, const Matrix& A);

    ///--- Nested classes ---

    /// Nested class Element, reference to a position in the matrix
    class Element {
    public:
    
      Element(Matrix& matrix, unsigned int i, unsigned int j);
      
      operator real() const;
      
      void operator=  (real a);
      void operator=  (const Element& e);
      void operator+= (real a);
      void operator-= (real a);
      void operator*= (real a);
      void operator/= (real a);
      

    protected:
    
      Matrix& A;
      unsigned int i;
      unsigned int j;

    };

    /// Nested class Row, reference to a row in the matrix
    class Row {
    public:
    
      Row(Matrix& matrix, unsigned int i, Range j);
      Row(Matrix& matrix, Index i, Range j);

      unsigned int size() const;

      real max() const;
      real min() const;
      real sum() const;
      real norm(unsigned int type) const;

      real operator() (unsigned int j) const;
      Element operator()(unsigned int j);

      void operator= (const Row& row);
      void operator= (const Column& col);
      void operator= (const Vector& x);

      real operator* (const Vector& x) const;

      friend class Column;

    private:
    
      Matrix& A;
      unsigned int i;
      Range j;

    };

    /// Nested class Column, reference to a column in the matrix
    class Column {
    public:
    
      Column(Matrix& matrix, Range i, unsigned int j);
      Column(Matrix& matrix, Range i, Index j);

      unsigned int size() const;

      real max() const;
      real min() const;
      real sum() const;
      real norm(unsigned int type) const;

      real operator() (unsigned int i) const;
      Element operator()(unsigned int i);

      void operator= (const Column& col);
      void operator= (const Row& row);
      void operator= (const Vector& x);

      real operator* (const Vector& x) const;

      friend class Row;

    private:
    
      Matrix& A;
      Range i;
      unsigned int j;

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
    unsigned int* permutation();
    
    /// Return permutation (only dense)
    unsigned int* const permutation() const;
    
    GenericMatrix* A;
    Type _type;

  };
  
}

#endif
