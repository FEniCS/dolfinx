// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-11-02
// Last changed: 2005-11-09

#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>

namespace dolfin
{

  /// Controller for adaptive time step selection, based on the list
  /// of controllers presented in "Digital Filters in Adaptive
  /// Time-Stepping" by Gustaf Soderlind (ACM TOMS 2003).

  class Controller
  {
  public:
    
    /// Create uninitialized controller
    Controller();

    /// Create controller with given initial state
    Controller(real k, real tol, uint p, real kmax);

    /// Destructor
    ~Controller();

    /// Initialize controller
    void init(real k, real tol, uint p, real kmax);

    /// Reset controller
    void reset(real k);

    /// Default controller
    real update(real e, real tol);

    /// Controller H0211
    real updateH0211(real e, real tol);

    /// Controller H211PI
    real updateH211PI(real e, real tol);

    /// No control, simple formula
    real updateSimple(real e, real tol);

    /// Control by harmonic mean value
    real updateHarmonic(real e, real tol);

    /// Control by harmonic mean value (no history supplied)
    static real updateHarmonic(real knew, real kold, real kmax);

  private:

    // Time step history
    double k0, k1;

    // Error history
    double e0;

    // Asymptotics: e ~ k^p
    double p;

    // Maximum time step
    double kmax;

  };

}

#endif
