// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __NEW_VECTOR_H
#define __NEW_VECTOR_H

#include <dolfin/GenericMatrix.h>
#include <dolfin/dolfin_main.h>

#include <dolfin/default_la_types.h>

namespace dolfin
{

  class NewVector : public GenericVector, public Variable
  {
    /// This class defines an interface for a Vector. The underlying vector 
    /// is defined in default_type.h.
    
    public:

      NewVector(){}

      ~NewVector() {}

      void init(uint i) 
        { vector.init(i); }

      uint size() const
        { return vector.size(); }

      /// Get values
      void get(real* values) const
        { vector.get(values); }

      /// Set values
      void set(real* values)
        { vector.set(values); }

      /// Add values
      void add(real* values)
        { vector.add(values); }

      /// Get block of values
      void get(real* block, uint m, const uint* rows) const
        { vector.get(block, m, rows); }

      /// Set block of values
      void set(const real* block, uint m, const uint* rows)
        { vector.set(block, m, rows); }

      /// Add block of values
      void add(const real* block, const uint m, const uint* rows)
        { vector.add(block, m, rows); }

      /// Set all entries to zero
      void zero()
        { vector.zero(); }

      /// Apply changes to matrix
      void apply()
        { vector.apply(); }

      /// Display matrix (sparse output is default)
      void disp(uint precision = 2) const
        { vector.disp(precision); }

      /// Return underlying vector (const version)
      const DefaultVector& vec() const
        { return vector; }

      /// Return underlying vector
      DefaultVector& vec()
        { return vector; }

    private:

      DefaultVector vector;
  };
}
#endif
