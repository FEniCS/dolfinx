// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITY_SOLVER_H
#define __ELASTICITY_SOLVER_H

#include <dolfin/NewSolver.h>

namespace dolfin {
  
  class ElasticitySolver : public NewSolver
  {
  public:
    
    // Create elasticity solver
    ElasticitySolver(Mesh& mesh,
		     NewFunction& f, NewFunction& u0, NewFunction& v0,
		     real E, real nu,
		     NewBoundaryCondition& bc, real k, real T);
    
    // Solve elasticity
    void solve();
    
    void save(Mesh& mesh, NewFunction& u, NewFunction& v, File &solutionfile);

    // Solve elasticity (static version)
    static void solve(Mesh& mesh,
		      NewFunction& f, NewFunction& u0, NewFunction& v0,
		      real E, real nu,
		      NewBoundaryCondition& bc, real k, real T);
    
  private:
    
    Mesh& mesh;
    NewFunction& f;
    NewFunction& u0;
    NewFunction& v0;
    real E;
    real nu;
    NewBoundaryCondition& bc;
    real k;
    real T;
    int counter;

  };
  
}

#endif
