// Copyright (C) 2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __HEAT_SOLVER_H
#define __HEAT_SOLVER_H

#include <dolfin/NewSolver.h>

namespace dolfin
{

  class HeatSolver 
  {
  public:
    
    // Create Heat solver
    HeatSolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc, 
	       NewFunction& u0, real dt, real T0, real T);
    
    // Solve Heat equation
    void solve();

    // Solve Heat equation (static version)
    static void solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc, 
		      NewFunction& u0, real dt, real T0, real T);
  
  private:

    Mesh& mesh;
    NewFunction& f;
    NewBoundaryCondition& bc;
    NewFunction& u0;
    real dt;
    real T0;
    real T;

  };

}

#endif
