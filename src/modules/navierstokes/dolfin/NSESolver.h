// Copyright (C) 2005 Johan Hoffman 
// Licensed under the GNU GPL Version 2.

#ifndef __NSE_SOLVER_H
#define __NSE_SOLVER_H

#include <dolfin/NewSolver.h>

namespace dolfin
{

  class NSESolver 
  {
  public:
    
    // Create Navier-Stokes solver
    NSESolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc, NewFunction& u0);
    
    // Solve Navier-Stokes equations
    void solve();

    // Solve Navier-Stokes equations (static version)
    static void solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc, NewFunction& u0);
  
  private:

    Mesh& mesh;
    NewFunction& f;
    NewBoundaryCondition& bc;
    NewFunction& u0;

  };

}

#endif
