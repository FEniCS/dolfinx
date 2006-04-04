// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-03-04
// Last changed: 

#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H


#include <dolfin/Variable.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace dolfin
{

  /// This class is a wrapper for the matrix template from uBlas, which is
  /// part of the freely available Boost C++ library (www.boost.org). 
  /// DenseMatrix is intended to operate together with DenseVector. Further 
  /// information and a listing of member functions can be found at 
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  class DenseMatrix : public boost::numeric::ublas::matrix<double>, public Variable
  {
  public:
 
    /// Constructor
    DenseMatrix();
    
    /// Constructor
    DenseMatrix(uint i, uint j);

    /// Destructor
    ~DenseMatrix();

    // Compute inverse of matrix
    void invert();

  private:


  };
}

#endif
