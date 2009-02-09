// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-09
// Last changed: 2009-02-09

#ifndef __ODE_COLLECTION_H
#define __ODE_COLLECTION_H

#include "ODE.h"

namespace dolfin
{

  /// An ODECollection represents a collection of initial value
  /// problems of the form
  ///
  ///     u'(t) = f(u(t), t) on (0, T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.
  ///
  /// Each ODE is governed by the same equation but a separate
  /// state is maintained for each ODE. Using ODECollection is
  /// recommended when solving a large number of ODEs and the
  /// overhead of instantiating a large number of ODE objects
  /// should be avoided.

  class ODECollection : public ODE
  {
  public:

    /// Create a collection of n ODEs of size N with final time T
    ODECollection(uint n, uint N, real T);
    
    /// Destructor
    virtual ~ODECollection();

    /// Set initial values
    void u0(real* u);

  private:

    // Number of ODE systems
    uint n;

    // States vectors
    real* states;

  };

}

#endif
