// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __COMPLEX_ODE
#define __COMPLEX_ODE

#include <dolfin/constants.h>
#include <dolfin/ODE.h>

namespace dolfin
{

  /// A ComplexODE represents an initial value problem for a system of
  /// complex-valued ordinary differential equations:
  ///
  ///     M(z, t) z'(t) = f(z(t), t) on (0,T]
  ///
  ///     z(0) = z0,
  ///
  /// where z(t) is a complex-valued vector of length n.
  ///
  /// This class is a wrapper for a standard real-valued ODE, and
  /// provides an interface that automatically translates the given
  /// complex-valued ODE of size n to a standard real-valued ODE of
  /// size N = 2n.
  ///
  /// The real and imaginary parts of the solution are stored in the
  /// following order in the solution vector u(t):
  ///
  /// u = (Re z0, Im z0, Re z1, Im z1, ..., Re z_n-1, Im z_n-1).

  class ComplexODE : public ODE
  {
  public:

    /// Constructor
    ComplexODE(uint n);

    /// Destructor
    ~ComplexODE();
    
    /// Initial value
    virtual complex z0(uint i) = 0;

    /// Evaluate right-hand side (multi-adaptive version)
    virtual complex f(complex z[], real t, uint i);

    /// Evaluate right-hand side (mono-adaptive version)
    virtual void feval(complex z[], real t, complex f[]);

    /// Return time step for component i
    virtual real k(uint i);

    /// Return initial value for real-valued ODE
    real u0(uint i);

    /// Return right-hand side for real-valued ODE
    real f(real u[], real t, uint i);

    /// Evaluate right-hand side for real-valued ODE
    void feval(real u[], real t, real f[]);

    /// Return time step for real-valued ODE
    real timestep(uint i);

  protected:

    // Number of complex components
    unsigned int n;

    // Complex-valued solution vector
    complex* zvalues;

    // Complex-valued right-hand side
    complex* fvalues;

  };

}

#endif
