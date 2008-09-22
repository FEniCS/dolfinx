// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-11-28
// Last changed: 2008-06-18

#ifndef __DUAL_H
#define __DUAL_H

#include "ODE.h"
#include "ODESolution.h"
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
    Dual(ODE& primal, ODESolution& u); 

    /// Destructor
    ~Dual();

    /// Initial value
    void u0(uBLASVector& u);

    /// Right-hand side
    void f(const uBLASVector& phi, real t, uBLASVector& y);

    /// Return real time (might be flipped backwards for dual)
    real time(real t) const;

  private:

    ODE& primal;
    ODESolution& u;

  };

}  //end namespace dolfin

#endif

