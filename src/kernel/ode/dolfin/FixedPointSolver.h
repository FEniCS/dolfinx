// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_SOLVER_H
#define __FIXED_POINT_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class NewTimeSlab;
  class NewMethod;
  
  /// This class implements fixed point iteration on time slabs. In
  /// each iteration, the solution is updated according to the fixed
  /// point iteration x = g(x). The iteration is performed forward in
  /// time Gauss-Seidel style, i.e., the degrees of freedom on an
  /// element are updated according to x = g(x) and the new values are
  /// used when updating the remaining elements.

  class FixedPointSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    FixedPointSolver(ODE& ode, NewTimeSlab& timeslab, const NewMethod& method);

    /// Destructor
    ~FixedPointSolver();

    /// Solve system
    void solve();

  protected:

    // Make an iteration
    real iteration();

  private:

    real* f; // Values of right-hand side at quadrature points

  };

}

#endif
