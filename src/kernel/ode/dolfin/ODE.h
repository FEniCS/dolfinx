// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004, 2005.

#ifndef __ODE_H
#define __ODE_H

#include <dolfin/constants.h>
#include <dolfin/Event.h>
#include <dolfin/Dependencies.h>
#include <dolfin/NewSample.h>

namespace dolfin
{

  /// A ODE represents an initial value problem of the form
  ///
  ///     u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.

  class ODE
  {
  public:

    /// Constructor
    ODE(uint N);
    
    /// Destructor
    virtual ~ODE();

    /// Return initial value for given component
    virtual real u0(uint i) = 0;

    /// Evaluate right-hand side (multi-adaptive version)
    // FIXME:: make this abstract?
    virtual real f(const real u[], real t, uint i);

    /// Evaluate right-hand side (mono-adaptive version)
    virtual void f(const real u[], real t, real y[]);

    /// Compute product y = Mx for implicit system
    virtual void M(const real x[], real y[], const real u[], real t);

    /// Compute product y = Jx for Jacobian J
    virtual void J(const real x[], real y[], const real u[], real t);

    /// Jacobian (optional)
    virtual real dfdu(const real u[], real t, uint i, uint j);

    /// Time step to use for whole system (optional)
    virtual real timestep();
    
    /// Time step to use for given component (optional)
    virtual real timestep(uint i);

    /// Update ODE, return false to stop (optional)
    virtual bool update(const real u[], real t, bool end);

    /// Save sample (optional)
    virtual void save(NewSample& sample);

    /// Number of components N
    uint size() const;

    /// End time (final time)
    real endtime() const;

    /// Solve ODE
    void solve();

    /// Automatically detect sparsity
    void sparse();

    /// Compute sparsity from given matrix
    void sparse(const Matrix& A);

    /// Friends
    friend class Dual;
    friend class RHS;
    friend class NewTimeSlab;
    friend class TimeSlabJacobian;
    friend class MonoAdaptiveTimeSlab;
    friend class MonoAdaptiveJacobian;
    friend class MultiAdaptiveTimeSlab;
    friend class MultiAdaptiveJacobian;
    friend class MultiAdaptivePreconditioner;
    friend class ReducedModel;
    friend class JacobianMatrix;

  protected:
    
    // Number of components
    uint N;
    
    // Final time
    real T;
    
    // Dependencies
    Dependencies dependencies;

    // Transpose of dependencies
    Dependencies transpose;

    // Default time step
    real default_timestep;

  private:

    Event not_impl_f;
    Event not_impl_M;
    Event not_impl_J;

  };

}

#endif
