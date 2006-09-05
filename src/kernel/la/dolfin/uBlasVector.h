// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-03-04
// Last changed: 2006-09-05

#ifndef __UBLAS_VECTOR_H
#define __UBLAS_VECTOR_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/ublas.h>
#include <dolfin/GenericVector.h>

namespace dolfin
{

  namespace ublas = boost::numeric::ublas;
  typedef ublas::vector<double> ublas_vector;

  /// This class represents a dense vector of dimension N.
  /// It is a simple wrapper for a Boost ublas vector.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// refer to the documentation for ublas which can be found at
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  class uBlasVector : public GenericVector,
		      public Variable,
		      public ublas_vector
  {
  public:
    
    /// Constructor
    uBlasVector();
    
    /// Constructor
    uBlasVector(uint N);
    
    /// Constructor from a uBlas vector_expression
    template <class E>
    uBlasVector(const ublas::vector_expression<E>& x) : ublas_vector(x) {}

    /// Destructor
    ~uBlasVector();

    /// Initialize a vector of length N
    void init(uint N);

    /// Set all entries to a single scalar value
    const uBlasVector& operator= (real a);

    /// Assignment from a vector_expression
    template <class E>
    uBlasVector& operator=(const ublas::vector_expression<E>& A)
    { 
      ublas::vector<double>::operator=(A); 
      return *this;
    } 
    
    /// Return size
    inline uint size() const
    { return ublas::vector<real>::size(); }

    /// Access given entry
    inline real& operator() (uint i)
    { return ublas::vector<real>::operator() (i); };

    /// Access value of given entry
    inline real operator() (uint i) const
    { return ublas::vector<real>::operator() (i); };

    /// Access element value
    inline real get(uint i) const { return (*this)(i); }

    /// Set element value
    inline void set(uint i, real value) { (*this)(i) = value; }

    /// Set block of values
    void set(const real block[], const int pos[], int n);

    /// Add block of values
    void add(const real block[], const int pos[], int n);

    /// Get block of values
    void get(real block[], const int pos[], int n) const;

    /// Compute norm of vector
    enum NormType { l1, l2, linf };
    real norm(NormType type = l2) const;

    /// Compute sum of vector
    real sum() const;

    /// Apply changes to vector (dummy function for compatibility)
    void apply();

    /// Set all entries to zero
    void zero();
    
    /// Element-wise division
    void div(const uBlasVector& x);

    /// Display vector
    void disp(uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const uBlasVector& x);

    // Copy values between different vector representations
#ifdef HAVE_PETSC_H
    class PETScVector;
    void copy(const PETScVector& y);
#endif
    void copy(const uBlasVector& y);

  };
}

#endif
