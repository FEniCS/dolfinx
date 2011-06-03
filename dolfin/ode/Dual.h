// Copyright (C) 2003-2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-11-28
// Last changed: 2008-10-06

#ifndef __DUAL_H
#define __DUAL_H

#include "ODE.h"
#include "ODESolution.h"

namespace dolfin
{

  /// A Dual represents an initial value problem of the form
  ///
  ///   - phi'(t) = J(u, t)^* phi(t) on [0,T),
  ///
  ///     phi(T)  = psi,
  ///
  /// where phi(t) is a vector of length N, A is the Jacobian of the
  /// right-hand side f of the primal problem, and psi is given final
  /// time data for the dual.
  ///
  /// To solve the dual forward in time, it is rewritten using the
  /// substitution t -> T - t, i.e. we solve an initial value problem
  /// of the form
  ///
  ///     w'(t) = J(u(T-t), T-t)^* w(t) on (0,T],
  ///
  ///     w(0)  = psi,
  ///
  /// where w(t) = phi(T-t).

  class Dual : public ODE
  {
  public:

    /// Constructor
    Dual(ODE& primal, ODESolution& u);

    /// Destructor
    ~Dual();

    /// Initial value
    void u0(Array<real>& psi);

    /// Right-hand side
    void f(const Array<real>& phi, real t, Array<real>& y);

    /// Return real time (might be flipped backwards for dual)
    real time(real t) const;

  private:

    ODE& primal;
    ODESolution& u;

  };

}

#endif
