// Copyright (C) 2003-2005 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2008-06-18

#ifndef __ODE_SOLVER_H
#define __ODE_SOLVER_H

namespace dolfin
{

  class ODE;
  class ODESolution;

  /// Solves a given ODE of the form
  ///
  ///     u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.
  
  class ODESolver {
  public:

     //solve ODE
    static void solve(ODE& ode);
    //solve ODE and return ODESolution object
    static void solve(ODE& ode, ODESolution& u); 

  private:

    static void solvePrimal(ODE& ode, ODESolution& u);
    static void solveDual(ODE& ode, ODESolution& u);

  };

}

#endif
