// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DUAL_H
#define __DUAL_H

#include <dolfin/ODE.h>

namespace dolfin {

  /// A Dual represents an initial value problem of the form
  ///
  ///   - phi'(t) = J(u,t)^* phi(t) on [0,T),
  ///         
  ///     phi(T)  = psi,
  ///
  /// where phi(t) is a vector of length N, A is the Jacobian of the
  /// right-hand side f of the primal problem, and psi is given final
  /// time data for the dual.
  ///
  /// To solve the Dual forward in time, it is rewritten using the
  /// substitution t -> T - t, i.e. we solve an initial value problem
  /// of the form
  ///
  ///     w'(t) = J(u(T-t),T-t)^* w(t) on (0,T],
  ///         
  ///     w(0)  = psi,
  ///
  /// where w(t) = phi(T-t).

  class Dual : public ODE
  {
  public:

    /// Constructor
    Dual(ODE& ode, Function& u);

    /// Destructor
    ~Dual();

    /// Initial value
    real u0(unsigned int i);

    /// Right-hand side
    real f(const Vector& u, real t, unsigned int i);

  private:

    ODE& ode;
    Function& u;

  };

}

#endif
