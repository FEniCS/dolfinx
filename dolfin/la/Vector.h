// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2008.
// Modified by Kent-Andre Mardal 2008.
// Modified by Ola Skavhaug 2008.
// Modified by Martin Alnæs 2008.
//
// First added:  2007-07-03
// Last changed: 2008-04-11

#ifndef __VECTOR_H
#define __VECTOR_H

#include "GenericMatrix.h"
#include <dolfin/main/dolfin_main.h>

#include "default_la_types.h"
#include "VectorNormType.h"

namespace dolfin
{

  /// This class provides an interface to the default DOLFIN
  /// vector implementation as decided in default_la_types.h.

  class Vector : public GenericVector, public Variable
  {
  public:

    /// Constructor
    Vector() : GenericVector(), Variable("x", "DOLFIN vector"),
               vector(0)
    {
      // TODO: use globally selected linear algebra factory to create new vector of any backend
      vector = new DefaultVector();
    }
    
    /// Constructor
    Vector(uint N) : GenericVector(), Variable("x", "DOLFIN vector"),
                     vector(0)
    {
      // TODO: use globally selected linear algebra factory to create new vector of any backend
      vector = new DefaultVector(N);
    }

    /// Destructor
    ~Vector()
    { delete vector; }

    /// Initialize vector of size N
    void init(uint N) 
    { vector->init(N); }

    /// Return copy of vector
    Vector* copy() const
    { Vector* x = new Vector(); delete x->vector; x->vector = vector->copy(); return x; }

    /// Return size
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
        
    /// Set all entries to zero
    void zero()
    { vector->zero(); }
    
    /// Apply changes
    void apply()
    { vector->apply(); }

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
    { return vector->axpy(a, x); }
    
    /// Return inner product 
    real inner(const GenericVector& x) const
    { return vector->inner(x); }

    /// Compute norm of vector
    real norm(VectorNormType type = l2) const
    { return vector->norm(type); }

    /// Multiply vector by given number
    const Vector& operator*= (real a)
    { *vector *= a; return *this; }

    /// Add given vector
    virtual const Vector& operator+= (const GenericVector& x)
    { axpy(1.0, x); return *this; }

    /// Subtract given vector
    virtual const Vector& operator-= (const GenericVector& x)
    { axpy(-1.0, x); return *this; }

    /// Divide vector by given number
    virtual const Vector& operator/= (real a)
    { *this *= 1.0 / a; return *this; }

    /// Assignment operator
    const GenericVector& operator= (const GenericVector& x)
    { *vector = x; return *this; }

    /// Assignment operator
    const Vector& operator= (const Vector& x)
    { *vector = *x.vector; return *this; }

    /// Display vector (sparse output is default)
    void disp(uint precision = 2) const
    { vector->disp(precision); }

    ///--- Special functions ---

    /// Get linear algebra backend factory
    LinearAlgebraFactory& factory() const
    { return vector->factory(); }

    ///--- Special functions, intended for library use only ---

    /// Return const GenericVector* (internal library use only!)
    virtual const GenericVector* instance() const 
    { return vector; }

    /// Return instance (non-const version)
    virtual GenericVector* instance() 
    { return vector; }

  private:
    
    GenericVector* vector;
    
  };

}
#endif
