// Copyright (C) 2006-2008 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
// Modified by Ola Skavhaug 2008.
// Modified by Martin Alnæs 2008.
//
// First added:  2006-03-04
// Last changed: 2008-04-11

#ifndef __UBLAS_VECTOR_H
#define __UBLAS_VECTOR_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Variable.h>
#include "ublas.h"
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"

#include "VectorNormType.h"

namespace dolfin
{

#ifdef HAS_PETSC
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
                      public Variable
  {
  public:

    /// Constructor
    uBlasVector();
    
    /// Constructor
    uBlasVector(uint N);
    
    /// Constructor from a uBlas vector_expression
    template <class E>
    uBlasVector(const ublas::vector_expression<E>& x) : x(x) {}

    /// Destructor
    ~uBlasVector();

    /// Initialize a vector of length N
    void init(uint N);

    /// Create uninitialized vector
    uBlasVector* create() const;

    /// Create copy of vector
    uBlasVector* copy() const;

    /// Set all entries to a single scalar value
    const uBlasVector& operator= (real a);

    /// Assignment of vector
    const uBlasVector& operator= (const GenericVector& x);

    /// Assignment of vector
    const uBlasVector& operator= (const uBlasVector& x);

    /// Add vector
    const uBlasVector& operator+= (const GenericVector& x);

    /// Subtract vector
    const uBlasVector& operator-= (const GenericVector& x);

    /// Multiply vector with scalar 
    const uBlasVector& operator *= (real a);

    /// Divide vector with scalar 
    const uBlasVector& operator /= (real a);

    /// Return concrete (const) uBlasVector instance
    virtual const uBlasVector* instance() const 
      { return this; }

    /// Return concrete uBlasVector instance
    virtual uBlasVector* instance() 
      { return this; }

    /// Assignment from a vector_expression
    //template <class E>
    //uBlasVector& operator=(const ublas::vector_expression<E>& x)
    //{ 
    //  this->ublas_vector::operator=(x); 
    //  return *this;
    //} 
    
    /// Return size
    inline uint size() const
      { return x.size(); }

    /// Access given entry
    inline real& operator() (uint i)
      { return x(i); };

    /// Access value of given entry
    inline real operator() (uint i) const
      { return x(i); };

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
    { return ublas::sum(x); }
    
    /// Addition (AXPY)
    void axpy(real a, const GenericVector& x);

    /// Scalar multiplication
    void mult(real const a);

    /// Inner product 
    real inner(const GenericVector& x) const;

    /// Element-wise division
    void div(const uBlasVector& x);

    /// Display vector
    void disp(uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const uBlasVector& x);

    /// Return backend factory
    LinearAlgebraFactory& factory() const;

    /// Return uBLAS ublas_vector reference
    const ublas_vector& vec() const;

    /// Return uBLAS ublas_vector reference
    ublas_vector& vec();

    // Copy values between different vector representations
#ifdef HAS_PETSC
    void copy(const PETScVector& y, uint off1, uint off2, uint len);
#endif
    void copy(const uBlasVector& y, uint off1, uint off2, uint len);

  private:

    // Underlying uBLAS vector object
    ublas_vector x;

  };

  LogStream& operator<< (LogStream& stream, const uBlasVector& x);
  

  inline bool is_uBlasVector(const GenericVector & gv)
  {
    const uBlasVector * v = dynamic_cast<const uBlasVector*>(gv.instance());
    return bool(v);
  }
  
  inline uBlasVector & as_uBlasVector(GenericVector & gv)
  {
    uBlasVector * v = dynamic_cast<uBlasVector*>(gv.instance());
    if(!v) error("Cannot convert GenericVector to uBlasVector.");
    return *v;
  }

  inline const uBlasVector & as_const_uBlasVector(const GenericVector & gv)
  {
    const uBlasVector * v = dynamic_cast<const uBlasVector*>(gv.instance());
    if(!v) error("Cannot convert GenericVector to uBlasVector.");
    return *v;
  }
  
  inline ublas_vector & as_ublas_vector(GenericVector & gv)
  {
    uBlasVector & v = as_uBlasVector(gv);
    return v.vec();
  }
  
  inline const ublas_vector & as_const_ublas_vector(const GenericVector & gv)
  {
    const uBlasVector & v = as_const_uBlasVector(gv);
    return v.vec();
  }

}

#endif
