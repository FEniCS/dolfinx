// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_H
#define __ODE_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>
#include <dolfin/Sparsity.h>
#include <dolfin/Dependencies.h>
#include <dolfin/Sample.h>
#include <dolfin/NewSample.h>
#include <dolfin/RHS.h>
#include <dolfin/Function.h>
#include <dolfin/Solution.h>
#include <dolfin/Adaptivity.h>

namespace dolfin
{
  class Vector;
  class Function;

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

    /// Initial value
    virtual real u0(uint i) = 0;

    /// Right-hand side (new version, will be made abstract)
    virtual real f(real u[], real t, uint i);

    /// Right-hand side (old version, will be removed)
    virtual real f(const Vector& u, real t, uint i);

    /// Jacobian (optional)
    virtual real dfdu(real u[], real t, uint i, uint j);

    /// Jacobian (old version, will be removed)
    virtual real dfdu(const Vector& u, real t, uint i, uint j);

    /// Method to use for given component (optional)
    virtual Element::Type method(uint i);

    /// Order to use for given component (optional)
    virtual uint order(uint i);

    /// Time step to use for given component (optional)
    virtual real timestep(uint i);

    /// Update ODE (optional)
    virtual void update(real u[], real t);

    /// Update ODE (old version, will be removed)
    virtual void update(RHS& f, Function& u, real t);

    /// Update ODE (old version, will be removed)
    virtual void update(Solution& u, Adaptivity& adaptivity, real t);

    /// Save sample (old version, will be removed)
    virtual void save(Sample& sample);

    /// Save sample (optional)
    virtual void save(NewSample& sample);

    /// Number of components N
    uint size() const;

    /// End time (final time)
    real endtime() const;

    /// Solve ODE
    void solve();

    /// Solve ODE
    void solve(Function& u);

    /// Solve ODE
    void solve(Function& u, Function& phi);
    
    /// Automatically detect sparsity
    void sparse();

    /// Compute sparsity from given matrix
    void sparse(const Matrix& A);

    /// Friends
    friend class Dual;
    friend class RHS;
    friend class NewTimeSlab;
    friend class NewJacobianMatrix;
    friend class PropagatingPreconditioner;
    friend class ReducedModel;
    friend class JacobianMatrix;

  protected:
    
    // Number of components
    uint N;
    
    // Final time
    real T;
    
    // Sparsity (old version, will be removed)
    Sparsity sparsity;

    // Dependencies
    Dependencies dependencies;

    // Transpose of dependencies
    Dependencies transpose;

    // Default time step
    real default_timestep;

  private:

    Element::Type default_method;
    uint default_order;

  };

}

#endif
