// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_METHODS_H
#define __CGQ_METHODS_H

#include <dolfin/Array.h>
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
    
    /// Return given cG(q) method
    const cGqMethod& operator() (int q) const;

    /// Initialize given cG(q) method
    void init(int q);

  private:

    Array<cGqMethod*> methods;

  };
  
  /// Table of cG(q) methods common to all elements
  extern cGqMethods cG;
  
}

#endif
