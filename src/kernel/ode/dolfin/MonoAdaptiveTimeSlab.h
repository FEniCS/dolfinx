// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MONO_ADAPTIVE_TIME_SLAB_H
#define __MONO_ADAPTIVE_TIME_SLAB_H

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/NewVector.h>
#include <dolfin/MonoAdaptivity.h>
#include <dolfin/NewTimeSlab.h>

namespace dolfin
{
  
  class ODE;
  class NewMethod;
  class TimeSlabSolver;
  
  /// This class represents a mono-adaptive time slab of length k = b - a
  /// holding the degrees of freedom for the solution of an ODE between
  /// two time levels a and b.

  class MonoAdaptiveTimeSlab : public NewTimeSlab
  {
  public:

    /// Constructor
    MonoAdaptiveTimeSlab(ODE& ode);

    /// Destructor
    ~MonoAdaptiveTimeSlab();
    
    /// Build time slab, return end time
    real build(real a, real b);

    /// Solve time slab system
    void solve();

    /// Shift time slab (prepare for next time slab)
    void shift();

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
    real* f;                   // Values of right-hand side at all quadrature points
    NewVector x;               // Degrees of freedom for the solution

  };

}

#endif
