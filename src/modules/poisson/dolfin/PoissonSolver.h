// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

//#include <dolfin/Solver.h>

namespace dolfin
{

  class Mesh;
  class NewVector;
  class NewMatrix;
  class NewFunction;
  class NewBoundaryCondition;
  
  class PoissonSolver 
  {
  public:
    
    // Create Poisson solver
    PoissonSolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc);
    
    // Solve Poisson's equation
    void solve();

    // Solve Poisson's equation (static version)
    static void solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    NewFunction& f;
    NewBoundaryCondition& bc;

  };

}

#endif
