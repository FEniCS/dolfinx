// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2008-06-11

#ifndef __MONO_ADAPTIVE_TIME_SLAB_H
#define __MONO_ADAPTIVE_TIME_SLAB_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include <dolfin/la/uBlasVector.h>
#include "MonoAdaptivity.h"
#include "TimeSlab.h"

namespace dolfin
{  

  class ODE;
  class Method;
  class TimeSlabSolver;
  
  /// This class represents a mono-adaptive time slab of length k = b - a
  /// holding the degrees of freedom for the solution of an ODE between
  /// two time levels a and b.

  class MonoAdaptiveTimeSlab : public TimeSlab
  {
  public:

    /// Constructor
    MonoAdaptiveTimeSlab(ODE& ode);

    /// Destructor
    ~MonoAdaptiveTimeSlab();
    
    /// Build time slab, return end time
    real build(real a, real b);

    /// Solve time slab system
    bool solve();

    /// Check if current solution can be accepted
    bool check(bool first);

    /// Shift time slab (prepare for next time slab)
    bool shift(bool end);

    /// Prepare sample at time t
    void sample(real t);

    /// Sample solution value of given component at given time
    real usample(uint i, real t);

    /// Sample time step size for given component at given time
    real ksample(uint i, real t);

    /// Sample residual for given component at given time
    real rsample(uint i, real t);

    /// Display time slab data
    void disp() const;
    
    /// Friends
    friend class MonoAdaptiveFixedPointSolver;
    friend class MonoAdaptiveNewtonSolver;
    friend class MonoAdaptiveJacobian;

  private:

    // Evaluate right-hand side at given quadrature point
    void feval(uint m);
    
    // Choose solver
    TimeSlabSolver* chooseSolver();

    // Temporary data array used to store multiplications
    real* tmp();

    TimeSlabSolver* solver;    // The solver
    MonoAdaptivity adaptivity; // Adaptive time step regulation
    uint nj;                   // Number of dofs
    real* dofs;                // Local dofs for an element used for interpolation
    real* fq;                  // Values of right-hand side at all quadrature points
    real rmax;                 // Previously computed maximum norm of residual

    uBlasVector x; // Degrees of freedom for the solution on the time slab
    uBlasVector u; // The solution at a given stage
    uBlasVector f; // The right-hand side at a given stage
    
  };

}

#endif
