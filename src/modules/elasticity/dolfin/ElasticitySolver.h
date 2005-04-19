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
    ElasticitySolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc);
    
    // Solve elasticity
    void solve();
    
    // Solve elasticity (static version)
    static void solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc);
    
  private:
    
    Mesh& mesh;
    NewFunction& f;
    NewBoundaryCondition& bc;
    
  };
  
}

#endif
