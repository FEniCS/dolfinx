// Copyright (C) 2003-2005 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003
// Last changed: 2005

#ifndef __ODE_SOLVER_H
#define __ODE_SOLVER_H

#include <dolfin/main/constants.h>

namespace dolfin {

  class ODE;

  /// Solves a given ODE of the form
  ///
  ///     u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.
  
  class ODESolver {
  public:

    static void solve(ODE& ode);
    //static void solve(ODE& ode, Function& u);
    //static void solve(ODE& ode, Function& u, Function& phi);

  private:

    //static void solvePrimal(ODE& ode, Function& u);
    //static void solveDual(ODE& ode, Function& u, Function& phi);

  };

}

#endif
