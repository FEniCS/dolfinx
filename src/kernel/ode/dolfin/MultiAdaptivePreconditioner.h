// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PROPAGATING_PRECONDITIONER_H
#define __PROPAGATING_PRECONDITIONER_H

#include <dolfin/NewPreconditioner.h>

namespace dolfin
{

  class ODE;
  class NewVector;
  class NewMethod;
  class MultiAdaptiveTimeSlab;
  class MultiAdaptiveJacobian;
  
  /// This class implements a preconditioner for the Newton system to
  /// be solved on a multi-adaptive time slab. The preconditioner does
  /// Gauss-Seidel type fixed point iteration forward in time using
  /// diagonally scaled dG(0), and is responsible for propagating the
  /// values forward in time in each GMRES iteration.

  class MultiAdaptivePreconditioner : public NewPreconditioner
  {
  public:

    /// Constructor
    MultiAdaptivePreconditioner(const MultiAdaptiveJacobian& A);

    /// Destructor
    ~MultiAdaptivePreconditioner();
    
    /// Solve linear system approximately for given right-hand side b
    void solve(NewVector& x, const NewVector& b);

  private:

    /// The Jacobian of the time slab system
    const MultiAdaptiveJacobian& A;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // The ODE
    ODE& ode;
    
    // Method, mcG(q) or mdG(q)
    const NewMethod& method;

  };

}

#endif
