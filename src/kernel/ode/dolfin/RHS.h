// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __RHS_H
#define __RHS_H

#include <dolfin/constants.h>
#include <dolfin/Vector.h>

namespace dolfin {

  class ODE;
  class Solution;
  class Function;

  /// RHS takes care of evaluating the right-hand side f(u,t)
  /// for a given component at a given time. The vector u is
  /// updated only for the components which influence the
  /// given component, as determined by the sparsity pattern.

  class RHS {
  public:

    /// Constructor
    RHS(ODE& ode, Solution& solution);

    /// Constructor
    RHS(ODE& ode, Function& function);

    /// Destructor
    ~RHS();

    /// Number of components
    unsigned int size() const;
    
    /// Evaluation of the right-hand side
    real operator() (unsigned int index, unsigned int node, real t);

    // Compute derivative dfi/duj
    real dFdU(unsigned int i, unsigned int j, real t);

  private:

    // Update components that influence the current component at time t
    void update(unsigned int index, unsigned int node, real t);

    // Update components that influence the current component at time t
    void update(unsigned int index, real t);

    // The ODE
    ODE& ode;

    // Solution
    Solution* solution;

    // Function
    Function* function;

    // Solution vector
    Vector u;

  };
    
}

#endif
