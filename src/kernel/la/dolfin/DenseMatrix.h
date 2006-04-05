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


  // Test base class to examine overhead 
  class GenericMatrix 
  {
  public:
 
    /// Constructor
    GenericMatrix(){};

    /// Destructor
    virtual ~GenericMatrix(){};

    /// Access elements
//    virtual real& operator() (uint i, uint j) = 0; 
    virtual real& operator() (uint i, uint j)
     {
        dolfin_error("Shouldn;t be in GenericMatrix virtual function");
        real temp=0.0; real& junk = temp; return junk;
     }; 

  private:

  };

  // This class operates indepently of other DOLFIN matrix classes
  class DenseMatrix : public boost::numeric::ublas::matrix<real>, public Variable
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

//    boost::numeric::ublas::matrix<real>& operator= (DenseMatrix& A)
//    { 
//      return boost::numeric::ublas::matrix<real>::operator = (A) ; 
//    }; 

  private:


  };

  // Test class which derives from GenericMatrix
  class DenseMatrixDerived : public GenericMatrix, public boost::numeric::ublas::matrix<real>,  public Variable
  {
  public:

    using boost::numeric::ublas::matrix<real>::operator();
    
    /// Constructor
    DenseMatrixDerived();
    
    /// Constructor
    DenseMatrixDerived(uint i, uint j);

    /// Destructor
    ~DenseMatrixDerived();

    // Compute inverse of matrix
    void invert();

    /// Return address of an element 
      // This is necessary if real& operator() (uint i, uint j) is a pure virtual 
      // function in GenericMatrix, but has a cost of factor 2    
    real& operator() (uint i, uint j)
      { return boost::numeric::ublas::matrix<real>::operator() (i, j); }; 

    void copy(DenseMatrixDerived& A)
      { this->assign(A); }; 

    // Wrapping this function makes things faster. Why??
//    boost::numeric::ublas::matrix<real>& operator= (DenseMatrixDerived& A)
//    { 
//     return boost::numeric::ublas::matrix<real>::operator = (A) ; 
//    }; 

  private:


  };
  // Test base class to examine overhead for envelope-letter design 
  class NewMatrix 
  {
  public:
 
    /// Constructor
    NewMatrix(){ matrix = new DenseMatrixDerived; };

    /// Constructor
    NewMatrix(uint i, uint j){ matrix = new DenseMatrixDerived(i,j); };

    /// Destructor
    virtual ~NewMatrix(){};

    /// Access elements
//    virtual real& operator() (uint i, uint j) = 0; 
    real& operator() (uint i, uint j)
     {
        return (*matrix)(i,j);
     }; 

  private:
    
    GenericMatrix* matrix;

  };


}

#endif
