// Copyright (C) 2003-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2009-02-09

#ifndef __ODE_SOLVER_H
#define __ODE_SOLVER_H

namespace dolfin
{

  class ODE;
  class ODESolution;

  /// Solves a given ODE of the form
  ///
  ///     u'(t) = f(u(t), t) on [0, T],
  ///
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.

  class ODESolver
  {
  public:

    /// Create ODE solver for given ODE
    ODESolver(ODE& ode);

    /// Destructor
    ~ODESolver();

    // Solve ODE on [0, T]
    void solve();

    // Solve ODE on [0, T]
    void solve(ODESolution& u);

  private:

    // Solve primal problem
    void solve_primal(ODESolution& u);

    // The ODE
    ODE& ode;

  };

}

#endif
