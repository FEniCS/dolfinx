// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_H
#define __ODE_H

#include <dolfin/constants.h>
#include <dolfin/Element.h>
#include <dolfin/Sparsity.h>

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
    ODE(unsigned int N);
    
    /// Destructor
    virtual ~ODE();

    /// Initial value
    virtual real u0(unsigned int i) = 0;

    /// Right-hand side
    virtual real f(const Vector& u, real t, unsigned int i) = 0;

    /// Method to use for given component (optional)
    virtual Element::Type method(unsigned int i);

    /// Order to use for given component (optional)
    virtual unsigned int order(unsigned int i);

    /// Time step to use for given component (optional)
    virtual real timestep(unsigned int i);

    /// Number of components N
    unsigned int size() const;

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

    /// Friends
    friend class Dual;
    friend class RHS;

  protected:
    
    // Number of components
    unsigned int N;
    
    // Final time
    real T;
    
    // Sparsity
    Sparsity sparsity;

  private:

    Element::Type default_method;
    unsigned int default_order;
    real default_timestep;

  };

}

#endif
