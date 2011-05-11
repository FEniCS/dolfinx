// Copyright (C) 2003-2009 Johan Jansson and Anders Logg
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
