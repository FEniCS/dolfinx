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
  class NewBoundaryCondition;
  
  class PoissonSolver 
  {
  public:
    
    // Create Poisson solver
    PoissonSolver(Mesh& mesh, NewBoundaryCondition& bc);
    
    // Solve Poisson's equation
    void solve();

    // Solve Poisson's equation (static version)
    static void solve(Mesh& mesh, NewBoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    NewBoundaryCondition& bc;

    // FIXME: Remove when working
    void solveOld();
    void dirichletBC( NewMatrix& A, NewVector& b, Mesh& mesh);

  };

}

#endif
