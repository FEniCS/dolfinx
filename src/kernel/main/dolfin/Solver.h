// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_H
#define __SOLVER_H

#include <dolfin.h>

namespace dolfin {

  class Solver {
  public:

    /// Constructor for general solver
    Solver();

    /// Constructor for solver to problem on a given grid
    Solver(Grid& grid_);

    /// Constructor for ODE solver
    Solver(ODE& ode_);
    
    /// Problem description
    virtual const char* description() = 0;
    
    /// Solve problem
    virtual void solve() = 0;
    
  protected:
    
    Grid& grid;
    ODE& ode;

  private:
    
    // Every solver needs to have both a grid and an ODE (since the same solver
    // interface is used for different equations). We create dummy objects to
    // be able to initialize the references if they are not used.

    class DummyODE : public ODE {
    public:
      DummyODE() : ODE(0) {}
      real f(const Vector& u, real t, int i) { return 0.0; }
    };

    Grid     dummy_grid;
    DummyODE dummy_ode; 
    
  };
  
}

#endif
