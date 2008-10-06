// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2008-06-11

#ifndef __MONO_ADAPTIVE_TIME_SLAB_H
#define __MONO_ADAPTIVE_TIME_SLAB_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include <dolfin/la/uBLASVector.h>
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
    double build(double a, double b);

    /// Solve time slab system
    bool solve();

    /// Check if current solution can be accepted
    bool check(bool first);

    /// Shift time slab (prepare for next time slab)
    bool shift(bool end);

    /// Prepare sample at time t
    void sample(double t);

    /// Sample solution value of given component at given time
    double usample(uint i, double t);

    /// Sample time step size for given component at given time
    double ksample(uint i, double t);

    /// Sample residual for given component at given time
    double rsample(uint i, double t);

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
    double* tmp();

    TimeSlabSolver* solver;    // The solver
    MonoAdaptivity adaptivity; // Adaptive time step regulation
    uint nj;                   // Number of dofs
    double* dofs;                // Local dofs for an element used for interpolation
    double* fq;                  // Values of right-hand side at all quadrature points
    double rmax;                 // Previously computed maximum norm of residual

    uBLASVector x; // Degrees of freedom for the solution on the time slab
    uBLASVector u; // The solution at a given stage
    uBLASVector f; // The right-hand side at a given stage
    
  };

}

#endif
