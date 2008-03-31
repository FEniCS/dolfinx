// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007.
// Modified by Kent-Andre Mardal 2008.
//
// First added:  2007-07-03
// Last changed: 2008-03-21

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
    Vector() : GenericVector(), Variable("x", "DOLFIN vector") {}
    
    /// Constructor
    Vector(uint N) : GenericVector(), Variable("x", "DOLFIN vector"), vector(N) {}
    
    /// Destructor
    ~Vector() {}
    
    /// Initialize vector of size N
    inline void init(uint N) 
    { vector.init(N); }

    /// Create uninitialized vector
    inline Vector* create() const
    { return new Vector(); }

    /// Create copy of vector
    inline Vector* copy() const
    { error("Not yet implemented."); return 0; }

    /// Return size
    inline uint size() const
    { return vector.size(); }

    /// Get values
    inline void get(real* values) const
    { vector.get(values); }
    
    /// Set values
    inline void set(real* values)
    { vector.set(values); }
    
    /// Add values
    inline void add(real* values)
    { vector.add(values); }
    
    /// Get block of values
    inline void get(real* block, uint m, const uint* rows) const
    { vector.get(block, m, rows); }
    
    /// Set block of values
    inline void set(const real* block, uint m, const uint* rows)
    { vector.set(block, m, rows); }

    /// Add block of values
    inline void add(const real* block, uint m, const uint* rows)
    { vector.add(block, m, rows); }
        
    /// Set all entries to zero
    inline void zero()
    { vector.zero(); }
    
    /// Apply changes to matrix
    inline void apply()
    { vector.apply(); }
    
    /// Display matrix (sparse output is default)
    inline void disp(uint precision = 2) const
    { vector.disp(precision); }
    
    /// FIXME: Functions below are not in the GenericVector interface.
    /// FIXME: Should these be removed or added to the interface?

    /// Add vector x
    inline const Vector& operator+= (const Vector& x)
    { vector += x.vec(); 
    return *this; 
    }
    
    /// Compute norm of vector
    inline real norm(VectorNormType type = l2) const
    { return vector.norm(type); }
   
    /// Return const implementation
    inline const DefaultVector& vec() const
    { return vector; }
    
    /// Return implementation
    inline DefaultVector& vec()
    { return vector; }
    
    /// Return backend factory
    inline LinearAlgebraFactory& factory() const
    { return vector.factory(); }

    /// inner product 
    inline real inner(const GenericVector& x_) const
    { 
      const Vector* x = dynamic_cast<const Vector*>(&x_);  
      if (!x) error("The vector needs to be a Vector"); 
      return this->vector.inner(x->vector); 
    }

    /// this += a*x  
    inline void add(const GenericVector& x_, real a = 1.0) 
    { 
      const Vector* x = dynamic_cast<const Vector*>(&x_);  
      if (!x) error("The vector needs to be a Vector"); 
      return this->add(x->vector, a); 
    }




  private:
    
    DefaultVector vector;
    
  };

}
#endif
