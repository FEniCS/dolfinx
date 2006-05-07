// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-03-04
// Last changed: 2006-05-07

#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <dolfin/DenseVector.h>
#include <dolfin/GenericMatrix.h>

namespace dolfin
{

  /// This class is based on the matrix template from uBlas, which is
  /// part of the freely available Boost C++ library (www.boost.org). 
  /// Some wrappers are provided to provide a uniform interface to different
  /// DOLFIN matrix classes.
  ///
  /// DenseMatrix is intended to operate together with DenseVector. Further 
  /// information and a listing of member functions can be found at 
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  namespace ublas = boost::numeric::ublas;

  class DenseMatrix : public GenericMatrix<DenseMatrix>, 
      public ublas::matrix<real>, public Variable
  {
    typedef ublas::matrix<double> BaseMatrix;

  public:

    /// Constructor
    DenseMatrix();
    
    /// Constructor
    DenseMatrix(uint M, uint N);

    /// Destructor
    ~DenseMatrix();

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    /// nzmax is a dummy argument ans is passed for compatibility with sparse matrces
    void init(uint M, uint N, uint nzmax);

    /// Assignment from a matrix_expression
    template <class E>
    DenseMatrix& operator=(const ublas::matrix_expression<E>& A) const
    { 
      BaseMatrix::operator=(A); 
      return *this;
    } 

    /// Return reference to matrix component
    real& operator() (uint i, uint j) 
      { return BaseMatrix::operator() (i, j); }; 

    /// Set all entries to zero (should really use DenseMatrix::clear())
    DenseMatrix& operator= (real zero);

    void clear()
      { this->BaseMatrix::clear(); }

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    uint size(uint dim) const;

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Compute inverse of matrix
    void invert();

    /// Dummy function for compatibility with sparse matrix
    inline void apply(){};

    /// Maximum number of non-zero terms on a row. Not relevant for dense matrix
    /// so just return 0
    uint nzmax() const
      { return 0; }

    /// Set given rows to identity matrix
    void ident(const int rows[], int m);

    /// Compute A*x
    void mult(const DenseVector& x, DenseVector& Ax) const;

    /// Display matrix
    void disp(uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const DenseMatrix& A);
    
  private:

  };
}

#endif
