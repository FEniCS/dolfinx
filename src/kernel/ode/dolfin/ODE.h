// Copyright (C) 2003-2006 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-10-21
// Last changed: 2006-07-05

#ifndef __ODE_H
#define __ODE_H

#include <dolfin/constants.h>
#include <dolfin/Event.h>
#include <dolfin/DenseVector.h>
#include <dolfin/Dependencies.h>
#include <dolfin/Sample.h>

namespace dolfin
{
  
  class uBlasSparseMatrix;

  /// A ODE represents an initial value problem of the form
  ///
  ///     u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N.
  ///
  /// To define an ODE, a user must create a subclass of ODE
  /// and define the function u0() containing the initial
  /// condition, as well the function f() containing the
  /// right-hand side.
  ///
  /// 
  /// It is also possible to solve implicit systems of the form
  ///
  ///     M(u(t), t) u'(t) = f(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// by setting the option "implicit" to true and defining the
  /// function M().

  class ODE
  {
  public:

    /// Constructor
    ODE(uint N, real T);
    
    /// Destructor
    virtual ~ODE();

    /// Return initial value for given component
    virtual real u0(uint i) = 0;

    /// Evaluate right-hand side y = f(u, t), mono-adaptive version
    virtual void f(const DenseVector& u, real t, DenseVector& y);

    /// Evaluate right-hand side f_i(u, t), multi-adaptive version
    virtual real f(const DenseVector& u, real t, uint i);

    /// Compute product y = Mx for implicit system (optional)
    virtual void M(const DenseVector& x, DenseVector& y, const DenseVector& u, real t);

    /// Compute product y = Jx for Jacobian J (optional)
    virtual void J(const DenseVector& x, DenseVector& y, const DenseVector& u, real t);

    /// Compute entry of Jacobian (optional)
    virtual real dfdu(const DenseVector& u, real t, uint i, uint j);

    /// Time step to use for whole system (optional)
    virtual real timestep(real t, real k0) const;
    
    /// Time step to use for given component (optional)
    virtual real timestep(real t, uint i, real k0) const;

    /// Update ODE, return false to stop (optional)
    virtual bool update(const DenseVector& u, real t, bool end);

    /// Save sample (optional)
    virtual void save(Sample& sample);

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
    DenseVector tmp;

    // Events
    Event not_impl_f;
    Event not_impl_M;
    Event not_impl_J;

  };

}

#endif
