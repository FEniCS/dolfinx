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

    /// Constructor for solver to problem on a given mesh
    Solver(Mesh& mesh_);

    /// Constructor for ODE solver
    Solver(ODE& ode_);
    
    /// Destructor
    virtual ~Solver();

    /// Problem description
    virtual const char* description() = 0;
    
    /// Solve problem
    virtual void solve() = 0;
    
  protected:
    
    Mesh& mesh;
    ODE& ode;

  private:
    
    // Every solver needs to have both a mesh and an ODE (since the same solver
    // interface is used for different equations). We create dummy objects to
    // be able to initialize the references if they are not used.

    class DummyODE : public ODE {
    public:
      DummyODE() : ODE(1) {}
      real f(const Vector& u, real t, unsigned int i) { return 0.0; }
    };

    Mesh     dummy_mesh;
    DummyODE dummy_ode; 
    
  };
  
}

#endif
