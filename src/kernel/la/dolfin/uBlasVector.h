// Copyright (C) 2006-2007 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2007.
//
// First added:  2006-03-04
// Last changed: 2007-05-15

#ifndef __UBLAS_VECTOR_H
#define __UBLAS_VECTOR_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/ublas.h>
#include <dolfin/GenericVector.h>

#include <dolfin/NormType.h>

namespace dolfin
{

#ifdef HAVE_PETSC_H
  class PETScVector;
#endif

  namespace ublas = boost::numeric::ublas;

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
      ublas_vector::operator=(A); 
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

    /// Get values
    void get(real* values) const;

    /// Set values
    void set(real* values);

    /// Add values
    void add(real* values);

    /// Get block of values
    void get(real* block, uint m, const uint* rows) const;

    /// Set block of values
    void set(const real* block, uint m, const uint* rows);

    /// Add block of values
    void add(const real* block, uint m, const uint* rows);

    /// Apply changes to vector (dummy function for compatibility)
    void apply();

    /// Set all entries to zero
    void zero();

    /// Compute norm of vector
    real norm(VectorNormType type = l2) const;

    /// Compute sum of vector
    real sum() const
    { return ublas::sum(*this); }
    
    /// Addition (AXPY)
    void axpy(real a, const uBlasVector& x);

    /// Scalar multiplication
    void mult(real a);

    /// Element-wise division
    void div(const uBlasVector& x);

    /// Display vector
    void disp(uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const uBlasVector& x);

    // Copy values between different vector representations
#ifdef HAVE_PETSC_H
    void copy(const PETScVector& y, uint off1, uint off2, uint len);
#endif
    void copy(const uBlasVector& y, uint off1, uint off2, uint len);

  };
}

#endif
