// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-04
// Last changed: 2005-11-04

#ifndef __ADAPTIVITY_H
#define __ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Controller.h>

namespace dolfin
{
  
  class ODE;
  class Method;

  /// This is the base class for MonoAdaptivity and MultiAdaptivity,
  /// collecting common functionality for adaptive time-stepping.

  class Adaptivity
  {
  public:

    /// Constructor
    Adaptivity(const ODE& ode, const Method& method);

    /// Destructor
    ~Adaptivity();

    /// Check if current solution can be accepted
    bool accept();

    /// Return threshold for reaching end of interval
    real threshold() const;

  protected:

    // The ODE
    const ODE& ode;
   
    // The method
    const Method& method;

    // Tolerance
    real tol;

    // Maximum allowed time step
    real kmax;

    // Threshold for reaching end of interval
    real beta;

    // Safety factor for tolerance
    real safety;

    // Previous safety factor for tolerance
    real safety_old;

    // Maximum allowed safety factor for tolerance
    real safety_max;

    // Total number of rejected time steps
    uint num_rejected;

    // True if we should accept the current solution
    bool _accept;

    // Flag for fixed time steps
    bool kfixed;
    
  };

}

#endif
