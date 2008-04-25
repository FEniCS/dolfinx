// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2007-07-03
// Last changed: 2008-04-23

#ifndef __VECTOR_H
#define __VECTOR_H

#include "default_la_types.h"
#include "GenericVector.h"

namespace dolfin
{

  /// This class provides the default DOLFIN vector class,
  /// based on the default DOLFIN linear algebra backend.

  class Vector : public GenericVector, public Variable
  {
  public:

    /// Create empty vector
    explicit Vector() : Variable("x", "DOLFIN vector"),
                        vector(new DefaultVector())
    {}
    
    /// Create vector of size N
    explicit Vector(uint N) : Variable("x", "DOLFIN vector"),
                              vector(new DefaultVector(N))
    {}

    /// Copy constructor
    explicit Vector(const Vector& x) : Variable("x", "DOLFIN vector"),
                                       vector(new DefaultVector((*x.vector).down_cast<DefaultVector>()))
    {}

    /// Destructor
    ~Vector()
    { delete vector; }

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    Vector* copy() const
    { Vector* x = new Vector(); delete x->vector; x->vector = vector->copy(); return x; }

    /// Set all entries to zero and keep any sparse structure
    void zero()
    { vector->zero(); }

    /// Finalize assembly of tensor
    void apply()
    { vector->apply(); }

    /// Display tensor
    void disp(uint precision=2) const
    { vector->disp(precision); }

    //--- Implementation of the GenericVector interface ---

    /// Initialize vector of size N
    void init(uint N) 
    { vector->init(N); }

    /// Return size of vector
    uint size() const
    { return vector->size(); }

    /// Get block of values
    void get(real* block, uint m, const uint* rows) const
    { vector->get(block, m, rows); }
 
    /// Set block of values
    void set(const real* block, uint m, const uint* rows)
    { vector->set(block, m, rows); }

    /// Add block of values
    void add(const real* block, uint m, const uint* rows)
    { vector->add(block, m, rows); }

    /// Get all values
    void get(real* values) const
    { vector->get(values); }

    /// Set all values
    void set(real* values)
    { vector->set(values); }

    /// Add values to each entry
    void add(real* values)
    { vector->add(values); }

    /// Add multiple of given vector (AXPY operation)
    void axpy(real a, const GenericVector& x)
    { vector->axpy(a, x); }

    /// Return inner product with given vector
    real inner(const GenericVector& x) const
    { return vector->inner(x); }

    /// Return norm of vector
    real norm(VectorNormType type=l2) const
    { return vector->norm(type); }

    /// Multiply vector by given number
    const Vector& operator*= (real a)
    { *vector *= a; return *this; }

    /// Assignment operator
    const GenericVector& operator= (const GenericVector& x)
    { *vector = x; return *this; }

    /// Assignment operator
    const Vector& operator= (const Vector& x)
    { *vector = *x.vector; return *this; }

    //--- Convenience functions ---

    /// Add given vector
    virtual const Vector& operator+= (const GenericVector& x)
    { axpy(1.0, x); return *this; }

    /// Subtract given vector
    virtual const Vector& operator-= (const GenericVector& x)
    { axpy(-1.0, x); return *this; }

    /// Divide vector by given number
    virtual const Vector& operator/= (real a)
    { *this *= 1.0 / a; return *this; }

    //--- Special functions ---

    /// Return linear algebra backend factory
    LinearAlgebraFactory& factory() const
    { return vector->factory(); }

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericVector* instance() const
    { return vector; }

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericVector* instance()
    { return vector; }

  private:

    // Pointer to concrete implementation
    GenericVector* vector;

  };

}

#endif
