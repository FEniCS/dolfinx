// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DGQ_METHODS_H
#define __DGQ_METHODS_H

#include <dolfin/ShortList.h>
#include <dolfin/dGqMethod.h>

namespace dolfin {

  /// A table / list of dG(q) methods. The purpose of this class is
  /// to have a global table of methods available to all elements.
  ///
  /// Example usage:
  ///
  ///     dG.init(q);
  ///     x0 = dG(q).point(0);
  ///     x1 = dG(q).point(1);

  class dGqMethods {
  public:

    /// Constructor
    dGqMethods();

    /// Destructor
    ~dGqMethods();
    
    /// Return given dG(q) method
    const dGqMethod& operator() (int q) const;

    /// Initialize given dG(q) method
    void init(int q);

  private:

    ShortList<dGqMethod*> methods;

  };
  
  /// Table of dG(q) methods common to all elements
  extern dGqMethods dG;
  
}

#endif
