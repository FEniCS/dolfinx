// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_METHODS_H
#define __CGQ_METHODS_H

#include <dolfin/cGqMethod.h>

namespace dolfin {

  /// A table / list of cG(q) methods. The purpose of this class is
  /// to have a global table of methods available to all elements.
  ///
  /// Example usage:
  ///
  ///     cG.init(q);
  ///     x0 = cG(q).point(0);
  ///     x1 = cG(q).point(1);

  class cGqMethods {
  public:

    /// Constructor
    cGqMethods();

    /// Destructor
    ~cGqMethods();
    
    /// Return given cG(q) method (inline optimized)
    inline const cGqMethod& operator() (unsigned int q) const
    {
      dolfin_assert(q >= 1);
      dolfin_assert(q < size);
      dolfin_assert(methods[q]);
      
      return *(methods[q]);
    }

    /// Initialize given cG(q) method
    void init(unsigned int q);

  private:
    
    cGqMethod** methods;
    unsigned int size;

  };
  
  /// Table of cG(q) methods common to all elements
  extern cGqMethods cG;
  
}

#endif
