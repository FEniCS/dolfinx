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

    /// Create uninitialized vector
    Vector* create() const
    { return new Vector(); }

    /// Create copy of vector
    Vector* copy() const
    { 
      Vector* v =  new Vector(vector->size()); 
      *v = *vector; 
      return v; 
    }

    /// Return size
    uint size() const
    { return vector->size(); }

    /// Get values
    void get(real* values) const
    { vector->get(values); }
    
    /// Set values
    void set(real* values)
    { vector->set(values); }
    
    /// Add values
    void add(real* values)
    { vector->add(values); }
    
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
    
    /// Apply changes to matrix
    void apply()
    { vector->apply(); }
    
    /// Display matrix (sparse output is default)
    void disp(uint precision = 2) const
    { vector->disp(precision); }
    
    /// FIXME: Functions below are not in the GenericVector interface.
    /// FIXME: Should these be removed or added to the interface?


    /// Assignment operator
    const Vector& operator= (const GenericVector& x)
    { 
      // get the underlying GenericVector instance (in case x is a Vector) 
      const GenericVector* y = x.instance();

      // employ the operator= of the underlying instance
      *vector = *y; 

      return *this; 
    }

    /// Assignment operator
    const Vector& operator= (const Vector& x)
    { 
      // employ the operator= of the underlying instance
      *vector = *(x.vector); 
      return *this; 
    }

    /// Compute norm of vector
    real norm(VectorNormType type = l2) const
    { return vector->norm(type); } // FIXME: This isn't in the GenericVector interface!
   
    /// Return backend factory
    LinearAlgebraFactory& factory() const
    { return vector->factory(); }

    /// inner product 
    real inner(const GenericVector& x) const
    { return vector->inner(x); }

    /// this += a*x  
    void axpy(real a, const GenericVector& x) 
    { return vector->axpy(a, x); }

    /// this *= a  
    void mult(const real a) 
    { return vector->mult(a); } // FIXME: This isn't in the GenericVector interface!

    /// Multiply vector by given number
    const Vector& operator*= (real a)
    { *vector *= a; return *this; }

    /// Return const GenericVector* (internal library use only!)
    virtual const GenericVector* instance() const 
    { return vector; }

    /// Return GenericVector* (internal library use only!)
    virtual GenericVector* instance() 
    { return vector; }

  private:
    
    //GenericVector* vector; // FIXME: Use this when above FIXME's have been handled.
    DefaultVector* vector;
    
  };

}
#endif
