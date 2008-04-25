// Copyright (C) 2006-2008 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2008.
// Modified by Kent-Andre Mardal 2008.
// Modified by Ola Skavhaug 2008.
// Modified by Martin Alnæs 2008.
//
// First added:  2006-03-04
// Last changed: 2008-04-22

#ifndef __UBLAS_VECTOR_H
#define __UBLAS_VECTOR_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Variable.h>
#include "ublas.h"
#include "GenericVector.h"

#include "VectorNormType.h"

namespace dolfin
{

#ifdef HAS_PETSC
  class PETScVector;
#endif

  namespace ublas = boost::numeric::ublas;
  
  //// Forward declarations
  class LinearAlgebraFactory;

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
    explicit uBlasVector(uint N);
    
    /// Constructor from a uBlas vector_expression
    template <class E>
    explicit uBlasVector(const ublas::vector_expression<E>& x) : x(x) {}

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

    /// Return size
    uint size() const
    { return x.size(); }

    /// Access value of given entry
    virtual real& operator[] (uint i)
    { return x(i); };

    /// Access value of given entry
    virtual real operator[] (uint i) const
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
    const ublas_vector& vec() const
    { return x; }

    /// Return uBLAS ublas_vector reference
    ublas_vector& vec()
    { return x; }

  private:

    // Underlying uBLAS vector object
    ublas_vector x;

  };

  LogStream& operator<< (LogStream& stream, const uBlasVector& x);
 
}

#endif
