// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-03-04
// Last changed: 2006-05-07

#ifndef __DENSE_VECTOR_H
#define __DENSE_VECTOR_H

#include <dolfin/Variable.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <dolfin/GenericVector.h>

namespace dolfin
{

  /// This class is a wrapper for the vector template from uBlas, which is
  /// part of the freely available Boost C++ library (www.boost.org). 
  /// DenseMatrix is intended to operate together with DenseMatrix. Further 
  /// information and a listing of member functions can be found at 
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.


  namespace ublas = boost::numeric::ublas;

  class DenseVector : public GenericVector<DenseVector>, 
      public ublas::vector<double>, public Variable
  {
    typedef ublas::vector<double> BaseVector;

  public:
 
    /// Constructor
    DenseVector();

    /// Constructor
    DenseVector(uint N);
    
    /// Constructor from a uBlas vector_expression
    template <class E>
    DenseVector(const ublas::vector_expression<E>& x) : BaseVector(x){}

    /// Copy constructor
//    DenseVector(const DenseVector& x);

    /// Destructor
    ~DenseVector();

    /// Initialize a vector of length N
    void init(uint N);

    /// Set all entries to a single scalar value
    const DenseVector& operator= (real a);

    /// Assignment from a vector_expression
    template <class E>
    DenseVector& operator=(const ublas::vector_expression<E>& A)
    { 
      ublas::vector<double>::operator=(A); 
      return *this;
    } 
    
//    // test
//    template <class E>
//    void operator=(const ublas::vector_expression<double>& x)
//    { 
//      ublas::vector<double>::operator=(x); 
//    } 

    void clear()
      { ublas::vector<real>::clear(); }

    /// Return size
    uint size() const
      { return ublas::vector<real>::size(); }; 

    /// Add block of values
    void add(const real block[], const int pos[], int n);

    /// Insert block of values
    void insert(const real block[], const int pos[], int n);

    /// Return reference to matrix component
    real& operator() (uint i) 
      { return ublas::vector<real>::operator() (i); }; 

    /// Dummy function for compatibility with sparse vector
    void apply(){};

    /// Display vector
    void disp(uint precision = 2) const;

  private:


  };
}

#endif
