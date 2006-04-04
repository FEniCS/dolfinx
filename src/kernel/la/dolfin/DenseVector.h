// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-03-04
// Last changed: 

#ifndef __DENSE_VECTOR_H
#define __DENSE_VECTOR_H

#include <dolfin/Variable.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace dolfin
{

  /// This class is a wrapper for the vector template from uBlas, which is
  /// part of the freely available Boost C++ library (www.boost.org). 
  /// DenseMatrix is intended to operate together with DenseMatrix. Further 
  /// information and a listing of member functions can be found at 
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  class DenseVector : public boost::numeric::ublas::vector<double>, public Variable
  {
  public:
 
    /// Constructor
    DenseVector();

    /// Constructor
    DenseVector(uint i);
    
    /// Constructor
    DenseVector(DenseVector& x);

    /// Destructor
    ~DenseVector();

  private:


  };
}

#endif
