// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_SOLVER_H
#define __ODE_SOLVER_H

#include <dolfin/constants.h>

namespace dolfin {

  class ODE;

  /// Solves a given ODE of the form
  ///
  ///     u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N, using one
  /// of (or a combination of) the multi-adaptive
  /// Galerkin methods mcG(q) or mdG(q).

  class ODESolver {
  public:

    static void solve(ODE& ode);

  private:

  };

}

#endif
