// Copyright (C) 2009 Anders Logg and Ola Skavhaug
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet 2009
//
// First added:  2009-02-09
// Last changed: 2009-09-10

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
    ODECollection(ODE& ode, uint num_systems);

    /// Destructor
    virtual ~ODECollection();

    /// Solve ODE collection on [t0, t1]
    void solve(real t0, real t1);

    /// Set state for given ODE system
    void set_state(uint system, const Array<real>& u);

    /// Set states for all ODE systems
    void set_state(const Array<real>& u);

    /// Get state for given ODE system
    void get_state(uint system, Array<real>& u);

    /// Get states for all ODE systems
    void get_state(Array<real>& u);

    /// Optional user-defined update, called between solves
    virtual void update(Array<real>& u, real t, uint system);

  private:

    // The ODE
    ODE& ode;

    // Number of ODE systems
    uint num_systems;

    // States vectors
    real* states;

  };

}

#endif
