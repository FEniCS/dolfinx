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
  
  //class PoissonSolver : public Solver
  class PoissonSolver 
  {
  public:
    
    PoissonSolver(Mesh& mesh);
    
    const char* description();
    void solve();
  
  private:

    Mesh& mesh;

    // FIXME: Remove when working
    void solveOld();
    void dirichletBC( NewMatrix& A, NewVector& b, Mesh& mesh);

  };

}

#endif
