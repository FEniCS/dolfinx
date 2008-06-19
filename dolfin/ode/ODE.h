// Copyright (C) 2003-2008 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2008-04-08

#ifndef __ODE_H
#define __ODE_H

#include <dolfin/common/types.h>
#include <dolfin/log/Event.h>
#include <dolfin/la/uBlasVector.h>
#include <dolfin/la/uBlasSparseMatrix.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/parameter/Parametrized.h>
#include "Dependencies.h"
#include "Sample.h"

namespace dolfin
{

  /// A ODE represents an initial value problem of the form
  ///
  ///     u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.
  ///
  /// To define an ODE, a user must create a subclass of ODE and
  /// create the function u0() defining the initial condition, as well
  /// the function f() defining the right-hand side.
  ///
  /// DOLFIN provides two types of ODE solvers: a set of standard
  /// mono-adaptive solvers with equal adaptive time steps for all
  /// components as well as a set of multi-adaptive solvers with
  /// individual and adaptive time steps for the different
  /// components. The right-hand side f() is defined differently for
  /// the two sets of methods, with the multi-adaptive solvers
  /// requiring a component-wise evaluation of the right-hand
  /// side. Only one right-hand side function f() needs to be defined
  /// for use of any particular solver.
  ///
  /// It is also possible to solve implicit systems of the form
  ///
  ///     M(u(t), t) u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// by setting the option "implicit" to true and defining the
  /// function M().

  class ODE : public Parametrized
  {
  public:

    /// Create an ODE of size N with final time T
    ODE(uint N, real T);
    
    /// Destructor
    virtual ~ODE();

    /// Set initial values
    virtual void u0(uBlasVector& u) = 0;

    /// Evaluate right-hand side y = f(u, t), mono-adaptive version (default, optional)
    virtual void f(const uBlasVector& u, real t, uBlasVector& y);

    /// Evaluate right-hand side f_i(u, t), multi-adaptive version (optional)
    virtual real f(const uBlasVector& u, real t, uint i);

    /// Compute product y = Mx for implicit system (optional)
    virtual void M(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t);

    /// Compute product y = Jx for Jacobian J (optional)
    virtual void J(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t);

    /// Compute product y = tranpose(J)x for Jacobian J (optional)
    /// Used when computing error estimate only
    virtual void JT(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t);

    /// Compute entry of Jacobian (optional)
    virtual real dfdu(const uBlasVector& u, real t, uint i, uint j);

    /// Time step to use for the whole system at a given time t (optional)
    virtual real timestep(real t, real k0) const;
    
    /// Time step to use for a given component at a given time t (optional)
    virtual real timestep(real t, uint i, real k0) const;

    /// Update ODE, return false to stop (optional)
    virtual bool update(const uBlasVector& u, real t, bool end);

    /// Save sample (optional)
    virtual void save(Sample& sample);

    /// Return real time (might be flipped backwards for dual)
    virtual real time(real t) const;

    /// Automatically detect sparsity (optional)
    void sparse();

    /// Compute sparsity from given matrix (optional)
    void sparse(const uBlasSparseMatrix& A);

    /// Return number of components N
    uint size() const;

    /// Return end time (final time T)
    real endtime() const;

    /// Solve ODE
    void solve();

    /// Friends
    friend class Dual;
    friend class RHS;
    friend class TimeSlab;
    friend class TimeSlabJacobian;
    friend class MonoAdaptiveTimeSlab;
    friend class MonoAdaptiveJacobian;
    friend class MultiAdaptiveTimeSlab;
    friend class MultiAdaptiveJacobian;
    friend class MultiAdaptivity;
    friend class NewMultiAdaptiveJacobian;
    friend class MultiAdaptivePreconditioner;
    friend class ReducedModel;
    friend class JacobianMatrix;
    friend class TimeStepper;

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

    // Temporary vector used for computing Jacobian
    uBlasVector tmp;

    // Events
    Event not_impl_f;
    Event not_impl_M;
    Event not_impl_J;
    Event not_impl_JT;
  };

}

#endif
