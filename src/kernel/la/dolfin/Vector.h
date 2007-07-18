// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007.
//
// First added:  2007-07-03
// Last changed: 2007-07-18

#ifndef __VECTOR_H
#define __VECTOR_H

#include <dolfin/PETScVector.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/dolfin_main.h>

#include <dolfin/default_la_types.h>
#include <dolfin/VectorNormType.h>

namespace dolfin
{

  /// This class defines the interface for a standard Vector using
  /// the default backend as decided in default_la_types.h.

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
    inline void add(const real* block, const uint m, const uint* rows)
    { vector.add(block, m, rows); }
    
    /// Compute norm of vector
    inline real norm(VectorNormType type = l2) const
    { return vector.norm(type); }
    
    /// Set all entries to zero
    inline void zero()
    { vector.zero(); }
    
    /// Add vector x
    inline const Vector& operator+= (const Vector& x)
    { vector += x.vec(); 
    return *this; 
    }
    
    /// Apply changes to matrix
    inline void apply()
    { vector.apply(); }
    
    /// Display matrix (sparse output is default)
    inline void disp(uint precision = 2) const
    { vector.disp(precision); }
    
    /// Return const implementation
    inline const DefaultVector& vec() const
    { return vector; }
    
    /// Return implementation
    inline DefaultVector& vec()
    { return vector; }
    
    private:

      DefaultVector vector;

  };

}
#endif
