// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2005

#ifndef __PROPAGATING_PRECONDITIONER_H
#define __PROPAGATING_PRECONDITIONER_H

#include <dolfin/Preconditioner.h>

namespace dolfin
{

  class ODE;
  class Vector;
  class Method;
  class MultiAdaptiveTimeSlab;
  class MultiAdaptiveJacobian;
  
  /// This class implements a preconditioner for the Newton system to
  /// be solved on a multi-adaptive time slab. The preconditioner does
  /// Gauss-Seidel type fixed point iteration forward in time using
  /// diagonally scaled dG(0), and is responsible for propagating the
  /// values forward in time in each GMRES iteration.

  class MultiAdaptivePreconditioner : public Preconditioner
  {
  public:

    /// Constructor
    MultiAdaptivePreconditioner(const MultiAdaptiveJacobian& A);

    /// Destructor
    ~MultiAdaptivePreconditioner();
    
    /// Solve linear system approximately for given right-hand side b
    void solve(Vector& x, const Vector& b);

  private:

    /// The Jacobian of the time slab system
    const MultiAdaptiveJacobian& A;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // The ODE
    ODE& ode;
    
    // Method, mcG(q) or mdG(q)
    const Method& method;

  };

}

#endif
