// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-09
// Last changed: 2009-02-10

#ifndef __ODE_COLLECTION_H
#define __ODE_COLLECTION_H

#include "ODESolution.h"
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

  class ODECollection
  {
  public:

    /// Create a collection of ODEs
    ODECollection(ODE& ode, uint n);
    
    /// Destructor
    virtual ~ODECollection();

    /// Solve ODE collection on [t0, t1]
    void solve(real t0, real t1);

    /// Set state for given ODE
    void set_state(uint i, const real* u);

    /// Set states for all ODEs
    void set_state(const real* u);

    /// Get state for given ODE
    void get_state(uint i, real* u);

    /// Get states for all ODEs
    void get_state(real* u);

  private:

    // The ODE
    ODE& ode;

    // The ODE solution
    ODESolution u;

    // Number of ODE systems
    uint n;

    // States vectors
    real* states;

  };

}

#endif
