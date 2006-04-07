// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-03-04
// Last changed: 2006-03-07

#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H


#include <dolfin/Variable.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

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

  class DenseMatrix : public boost::numeric::ublas::matrix<real>, public Variable
  {
  public:
 
    /// Constructor
    DenseMatrix();
    
    /// Constructor
    DenseMatrix(uint i, uint j);

    /// Destructor
    ~DenseMatrix();

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    /// nzmax is a dummy argument ans is passed for compatibility with sparse matrces
    void init(uint M, uint N, uint nzmax);


    // Important that this function is inlined for efficiency
    /// Return address of matrix component
    real& operator() (uint i, uint j) 
      { return boost::numeric::ublas::matrix<real>::operator() (i, j); }; 

//    boost::numeric::ublas::matrix<real>& operator= (DenseMatrix& A)
//    { 
//      return boost::numeric::ublas::matrix<real>::operator = (A) ; 
//    }; 

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    inline uint size(uint dim) const;

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Compute inverse of matrix
    void invert();

    /// Dummy function for compatibility with sparse matrix
    inline void apply(){};

    /// Display matrix
    void disp(uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const DenseMatrix& A);
    
  private:

  };
}

#endif
