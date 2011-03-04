// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-02-02
// Last changed: 2010-08-26

#ifndef __COMPLEX_ODE_H
#define __COMPLEX_ODE_H

#ifndef HAS_GMP

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include "ODE.h"

namespace dolfin
{

  /// A ComplexODE represents an initial value problem for a system of
  /// complex-valued ordinary differential equations:
  ///
  ///     M(z, t) z'(t) = f(z(t), t) on (0,T]
  ///
  ///     z(0) = z0,
  ///
  /// where z(t) is a complex-valued vector of length n. The imaginary
  /// unit is provided by the member variable j satisfying j^2 = -1.
  ///
  /// This class is a wrapper for a standard real-valued ODE, and
  /// provides an interface that automatically translates the given
  /// complex-valued ODE of size n to a standard real-valued ODE of
  /// size N = 2n.
  ///
  /// The double and imaginary parts of the solution are stored in the
  /// following order in the solution vector u(t):
  ///
  /// u = (Re z0, Im z0, Re z1, Im z1, ..., Re z_n-1, Im z_n-1).

  class ComplexODE : public ODE
  {
  public:

    /// Constructor
    ComplexODE(uint n, real T);

    /// Destructor
    ~ComplexODE();

    /// Set initial values
    virtual void z0(complex z[]) = 0;

    /// Evaluate right-hand side (multi-adaptive version)
    virtual complex f(const complex z[], real t, uint i);

    /// Evaluate right-hand side (mono-adaptive version)
    virtual void f(const complex z[], real t, complex y[]);

    /// Compute product y = Mx for implicit system
    virtual void M(const complex x[], complex y[], const complex z[], real t);

    /// Compute product y = Jx for Jacobian J
    virtual void J(const complex x[], complex y[], const complex u[], real t);

    /// Return time step for component i (optional)
    virtual real k(uint i);

    /// Update ODE, return false to stop (optional)
    virtual bool update(const complex z[], real t, bool end);

    /// Return initial value for real-valued ODE
    void u0(Array<real>& u);

    /// Return right-hand side for real-valued ODE
    real f(const Array<real>& u, real t, uint i);

    /// Evaluate right-hand side for real-valued ODE
    void f(const Array<real>& u, real t, Array<real>& y);

    /// Compute product y = Mx for real-valued ODE
    void M(const Array<real>& x, Array<real>& y, const Array<real>& u, real t);

    /// Compute product y = Jx for real-valued ODE
    void J(const Array<real>& x, Array<real>& y, const Array<real>& u, real t);

    /// Return time step for real-valued ODE
    real timestep(uint i);

    /// Update for real-valued ODE
    bool update(const Array<real>& u, real t, bool end);

  protected:

    // Number of complex components
    unsigned int n;

    // Imaginary unit
    complex j;

  private:

    // Complex-valued solution vector
    complex* zvalues;

    // Complex-valued right-hand side
    complex* fvalues;

    // Extra array for computing product y = Mx, initialized if needed
    complex* yvalues;

  };

}

#else

namespace dolfin
{
  /// Dummy implementation of ComplexODE used when DOLFIN is compiled
  /// with GMP support in which case ComplexODE is not available

  class DummyComplexODE : public ODE
  {
  public:

    DummyComplexODE(uint n, real T) : ODE(1, 0.0), j(0)
    {
      warning("ComplexODE not available when DOLFIN is compiled with GMP support.");
    }

    void u0(Array<real>& u) {}

    void f(const Array<real>& u, real t, Array<real>& y) {}

  protected:

    double j;

  };

  // Use typedef to not confuse documentation extraction scripts
  typedef DummyComplexODE ComplexODE;

}

#endif
#endif
