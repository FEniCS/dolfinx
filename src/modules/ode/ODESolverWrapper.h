// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_SOLVER_WRAPPER_H
#define __ODE_SOLVER_WRAPPER_H

#include <dolfin/Solver.h>

namespace dolfin {
  
  /// This is a wrapper for the ODE solver (Tanganyika), which is
  /// really not a module but is implemented in the kernel.
  ///
  /// Note that it is probably more convenient to call the ODE solver
  /// directly, rather than creating an Problem from a given ODE and
  /// then solve it:
  ///
  /// Calling the ODE solver directly (not using this module):
  ///
  ///     Lorenz lorenz;
  ///     lorenz.solve();
  ///
  /// Creating a Problem and then calling solve (using this module):
  ///
  ///     Lorenz lorenzODE;
  ///     Problem lorenz("ode", lorenzODE);
  ///     lorenz.solve();
  
  class ODESolverWrapper : public Solver {
  public:
    
    ODESolverWrapper(ODE& ode);
    
    const char* description();
    void solve();
    
  private:

    ODESolver solver;

  };

}

#endif
