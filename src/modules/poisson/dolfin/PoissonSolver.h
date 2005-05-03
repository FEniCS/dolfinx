// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Johan Jansson, 2005.
// Modified by Anders Logg, 2005.

#ifndef __POISSON_SOLVER_H
#define __POISSON_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{

  class PoissonSolver : public Solver
  {
  public:
    
    // Create Poisson solver
    PoissonSolver(Mesh& mesh, Function& f, BoundaryCondition& bc);
    
    // Solve Poisson's equation
    void solve();

    // Solve Poisson's equation (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc;

  };

}

#endif
