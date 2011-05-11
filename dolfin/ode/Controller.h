// Copyright (C) 2005 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
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
    real update_simple(real e, real tol);

    /// Control by harmonic mean value
    real update_harmonic(real e, real tol);

    /// Control by harmonic mean value (no history supplied)
    static real update_harmonic(real knew, real kold, real kmax);

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
