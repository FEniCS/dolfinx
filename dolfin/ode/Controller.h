// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-11-02
// Last changed: 2005-11-09

#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <dolfin/common/types.h>

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
    Controller(double k, double tol, uint p, double kmax);

    /// Destructor
    ~Controller();

    /// Initialize controller
    void init(double k, double tol, uint p, double kmax);

    /// Reset controller
    void reset(double k);

    /// Default controller
    double update(double e, double tol);

    /// Controller H0211
    double updateH0211(double e, double tol);

    /// Controller H211PI
    double updateH211PI(double e, double tol);

    /// No control, simple formula
    double updateSimple(double e, double tol);

    /// Control by harmonic mean value
    double updateHarmonic(double e, double tol);

    /// Control by harmonic mean value (no history supplied)
    static double updateHarmonic(double knew, double kold, double kmax);

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
