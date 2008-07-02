// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2007-07-03
// Last changed: 2008-05-17

#ifndef __VECTOR_H
#define __VECTOR_H

#include <dolfin/common/Variable.h>
#include "DefaultFactory.h"
#include "GenericVector.h"

namespace dolfin
{

  /// This class provides the default DOLFIN vector class,
  /// based on the default DOLFIN linear algebra backend.

  class Vector : public GenericVector, public Variable
  {
  public:

    /// Create empty vector
    Vector() : Variable("x", "DOLFIN vector"), vector(0)
    { DefaultFactory factory; vector = factory.createVector(); }

    /// Create vector of size N
    explicit Vector(uint N) : Variable("x", "DOLFIN vector"), vector(0)
    { DefaultFactory factory; vector = factory.createVector(); vector->init(N); }

    /// Copy constructor
    explicit Vector(const Vector& x) : Variable("x", "DOLFIN vector"),
                                       vector(x.vector->copy())
    {}

    /// Destructor
    virtual ~Vector()
    { delete vector; }

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    virtual Vector* copy() const
    { Vector* x = new Vector(); delete x->vector; x->vector = vector->copy(); return x; }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    { vector->zero(); }

    /// Finalize assembly of tensor
    virtual void apply(FinalizeType finaltype=FINALIZE) 
    { vector->apply(finaltype); }

    /// Display tensor
    virtual void disp(uint precision=2) const
    { vector->disp(precision); }

    //--- Implementation of the GenericVector interface ---

    /// Initialize vector of size N
    virtual void init(uint N) 
    { vector->init(N); }

    /// Return size of vector
    virtual uint size() const
    { return vector->size(); }

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows) const
    { vector->get(block, m, rows); }
 
    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows)
    { vector->set(block, m, rows); }

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows)
    { vector->add(block, m, rows); }

    /// Get all values
    virtual void get(real* values) const
    { vector->get(values); }

    /// Set all values
    virtual void set(real* values)
    { vector->set(values); }

    /// Add values to each entry
    virtual void add(real* values)
    { vector->add(values); }

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(real a, const GenericVector& x)
    { vector->axpy(a, x); }

    /// Return inner product with given vector
    virtual real inner(const GenericVector& x) const
    { return vector->inner(x); }

    /// Return norm of vector
    virtual real norm(VectorNormType type=l2) const
    { return vector->norm(type); }

    /// Return minimum value of vector
    virtual real min() const
    { return vector->min(); }

    /// Return maximum value of vector
    virtual real max() const
    { return vector->max(); }

    /// Multiply vector by given number
    virtual const Vector& operator*= (real a)
    { *vector *= a; return *this; }

    /// Divide vector by given number
    virtual const Vector& operator/= (real a)
    { *this *= 1.0 / a; return *this; }

    /// Add given vector
    virtual const Vector& operator+= (const GenericVector& x)
    { axpy(1.0, x); return *this; }

    /// Subtract given vector
    virtual const Vector& operator-= (const GenericVector& x)
    { axpy(-1.0, x); return *this; }

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x)
    { *vector = x; return *this; }

    /// Assignment operator
    const Vector& operator= (real a)
    { *vector = a; return *this; }

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const
    { return vector->factory(); }

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericVector* instance() const
    { return vector; }

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericVector* instance()
    { return vector; }

    //--- Special Vector functions ---

    /// Assignment operator
    const Vector& operator= (const Vector& x)
    { *vector = *x.vector; return *this; }

  private:

    // Pointer to concrete implementation
    GenericVector* vector;

  };

}

#endif
