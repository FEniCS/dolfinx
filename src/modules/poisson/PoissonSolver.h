// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{
  
  class PoissonSolver : public Solver
  {
  public:
    
    PoissonSolver(Mesh& mesh);
    
    const char* description();
    void solve();
  
  private:

    // FIXME: Remove when working
    void solveOld();
    void dirichletBC( NewMatrix& A, NewVector& b, Mesh& mesh);

  };

}

#endif
